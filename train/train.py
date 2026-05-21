"""AlphaZero self-play training loop for Bao la Kiswahili.

Run with default settings (M2-friendly):
    python -m train.train

Override any config field via CLI:
    python -m train.train num_simulations=32 use_wandb=true
"""

import datetime
import os
import pickle
import time
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
from omegaconf import OmegaConf
from pgx.experimental import auto_reset

from bao import Bao, NUM_ACTIONS
from train.config import Config
from train.network import AZNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

# ---------------------------------------------------------------------------
# Environment + network
# ---------------------------------------------------------------------------

env = Bao()


def forward_fn(x, is_eval: bool = False):
    net = AZNet(
        num_actions=NUM_ACTIONS,
        num_channels=config.num_channels,
        num_blocks=config.num_blocks,
    )
    return net(x, is_training=not is_eval)


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)

# ---------------------------------------------------------------------------
# MCTS transition model
# ---------------------------------------------------------------------------


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state):
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True
    )
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    # -1 when the player changed (zero-sum), +1 when the same player moves again
    # (nyumba_pending: same player must choose stop/continue).
    player_changed = state.current_player != current_player
    discount = jnp.where(state.terminated, 0.0, jnp.where(player_changed, -1.0, 1.0))

    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    ), state


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.jit
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size

    def step_fn(state, key):
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        player_changed = state.current_player != actor
        discount = jnp.where(state.terminated, 0.0, jnp.where(player_changed, -1.0, 1.0))
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(batch_size), actor],
            terminated=state.terminated,
            discount=discount,
        )

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)
    return data


# ---------------------------------------------------------------------------
# Loss input computation
# ---------------------------------------------------------------------------


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.jit
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size

    # mask[t] is True iff timestep t belongs to a completed episode
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Discounted return, scanned backwards: v[t] = r[t] + discount[t] * v[t+1]
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(body_fn, jnp.zeros(batch_size), jnp.arange(config.max_num_steps))
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


# ---------------------------------------------------------------------------
# Loss + training step
# ---------------------------------------------------------------------------


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )
    policy_loss = jnp.mean(optax.softmax_cross_entropy(logits, samples.policy_tgt))
    value_loss = jnp.mean(optax.l2_loss(value, samples.value_tgt) * samples.mask)
    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@jax.jit
def train_step(model, opt_state, batch: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, batch
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    return (model_params, model_state), opt_state, policy_loss, value_loss


# ---------------------------------------------------------------------------
# Evaluation: greedy model vs random opponent
# ---------------------------------------------------------------------------


@jax.jit
def evaluate(rng_key: jnp.ndarray, model) -> jnp.ndarray:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size
    my_player = 0

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)

    def step_fn(carry, key):
        state, R = carry
        already_done = state.terminated

        (logits, _), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        my_action = jnp.argmax(logits, axis=-1)

        # Uniform over legal actions for opponent
        random_logits = jnp.where(
            state.legal_action_mask, 0.0, jnp.finfo(jnp.float32).min
        )
        opp_action = jax.random.categorical(key, random_logits)

        is_my_turn = state.current_player == my_player
        action = jnp.where(is_my_turn, my_action, opp_action)

        state = jax.vmap(env.step)(state, action)
        # Only credit rewards from games that weren't already finished
        R = R + state.rewards[:, my_player] * (~already_done).astype(jnp.float32)
        return (state, R), None

    key_seq = jax.random.split(rng_key, config.max_num_steps)
    (_, R), _ = jax.lax.scan(step_fn, (state, jnp.zeros(batch_size)), key_seq)
    return R


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if config.use_wandb:
        import wandb
        wandb.init(project="bao-az", config=config.model_dump())

    # --- init model ---
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    model = forward.init(jax.random.PRNGKey(0), dummy_state.observation)  # (params, net_state)
    model_params, model_state = model
    opt_state = optimizer.init(model_params)

    # --- checkpoint dir ---
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"bao_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    rng_key = jax.random.PRNGKey(config.seed)
    iteration = 0
    frames = 0
    hours = 0.0
    log: dict = {"iteration": iteration, "hours": hours, "frames": frames}

    while True:
        if iteration % config.eval_interval == 0:
            rng_key, subkey = jax.random.split(rng_key)
            R = evaluate(subkey, model)
            log.update(
                {
                    "eval/vs_random/avg_R": R.mean().item(),
                    "eval/vs_random/win_rate": ((R > 0).sum() / R.size).item(),
                    "eval/vs_random/draw_rate": ((R == 0).sum() / R.size).item(),
                    "eval/vs_random/lose_rate": ((R < 0).sum() / R.size).item(),
                }
            )

            model_params, model_state = model
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                pickle.dump(
                    {
                        "config": config,
                        "rng_key": rng_key,
                        "model": jax.device_get(model),
                        "opt_state": jax.device_get(opt_state),
                        "iteration": iteration,
                        "frames": frames,
                        "hours": hours,
                    },
                    f,
                )

        print(log)
        if config.use_wandb:
            import wandb
            wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # --- self-play ---
        rng_key, subkey = jax.random.split(rng_key)
        data: SelfplayOutput = selfplay(model, subkey)
        samples: Sample = compute_loss_input(data)

        # Flatten (max_num_steps, batch) -> N, shuffle, minibatch
        samples = jax.device_get(samples)
        frames += samples.obs.shape[0] * samples.obs.shape[1]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), samples)

        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)

        N = samples.obs.shape[0]
        num_updates = N // config.training_batch_size
        if num_updates == 0:
            # batch smaller than training_batch_size; do one update on all data
            num_updates = 1
            minibatches = samples
        else:
            minibatches = jax.tree_util.tree_map(
                lambda x: x[: num_updates * config.training_batch_size].reshape(
                    (num_updates, config.training_batch_size, *x.shape[1:])
                ),
                samples,
            )

        # --- training ---
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            if num_updates == 1:
                batch = minibatches
            else:
                batch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, pl, vl = train_step(model, opt_state, batch)
            policy_losses.append(pl.item())
            value_losses.append(vl.item())

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": sum(policy_losses) / len(policy_losses),
                "train/value_loss": sum(value_losses) / len(value_losses),
                "hours": hours,
                "frames": frames,
            }
        )
