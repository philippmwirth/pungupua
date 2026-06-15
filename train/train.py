"""AlphaZero self-play training loop for Bao la Kiswahili.

Run with default settings (M2-friendly):
    python -m train.train

Override any config field via CLI:
    python -m train.train num_simulations=32 use_wandb=true
"""

import datetime
import glob
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import numpy as np

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
    return net(x, is_training=not is_eval, test_local_stats=False)


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

n_devices = jax.local_device_count()
_updates_per_iter = max(
    (n_devices * config.max_num_steps * config.selfplay_batch_size)
    // config.training_batch_size,
    1,
)
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=config.learning_rate,
    warmup_steps=config.learning_rate_warmup_steps,
    decay_steps=config.max_num_iters * _updates_per_iter,
    end_value=config.learning_rate_min,
)
optimizer = optax.adam(learning_rate=lr_schedule)

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


@jax.pmap
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
        state = jax.vmap(auto_reset(env.step, env.init))(
            state, policy_output.action, keys
        )
        player_changed = state.current_player != actor
        discount = jnp.where(
            state.terminated, 0.0, jnp.where(player_changed, -1.0, 1.0)
        )
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


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size

    # mask[t] is True iff timestep t belongs to a completed episode
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Discounted return, scanned backwards: v[t] = r[t] + discount[t] * v[t+1]
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn, jnp.zeros(batch_size), jnp.arange(config.max_num_steps)
    )
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
    policy_loss = jnp.mean(optax.softmax_cross_entropy(logits, samples.policy_tgt) * samples.mask)
    value_loss = jnp.mean(optax.l2_loss(value, samples.value_tgt) * samples.mask)
    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
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


@jax.pmap
def evaluate(rng_key: jnp.ndarray, model) -> tuple[jnp.ndarray, jnp.ndarray]:
    model_params, model_state = model
    batch_size = 1024
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
        return (state, R), (logits, state.terminated)

    key_seq = jax.random.split(rng_key, config.max_num_steps)
    (_, R), (all_logits, all_terminated) = jax.lax.scan(
        step_fn, (state, jnp.zeros(batch_size)), key_seq
    )
    return R, all_logits[:10], all_terminated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if config.use_wandb:
        import wandb

        wandb.init(project="pungupua", config=config.model_dump())

    # --- init or resume model ---
    if config.resume_from:
        ckpt_files = sorted(glob.glob(os.path.join(config.resume_from, "*.ckpt")))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {config.resume_from}")
        latest = ckpt_files[-1]
        print(f"Resuming from {latest}")
        with open(latest, "rb") as f:
            ckpt = pickle.load(f)
        model = ckpt["model"]
        opt_state = ckpt["opt_state"]
        rng_key = ckpt["rng_key"]
        iteration = ckpt["iteration"]
        frames = ckpt["frames"]
        hours = ckpt["hours"]
        grad_steps = ckpt.get("grad_steps", iteration * _updates_per_iter)
        ckpt_dir = config.resume_from
    else:
        dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
        model = forward.init(
            jax.random.PRNGKey(0), dummy_state.observation
        )  # (params, net_state)
        model_params, model_state = model
        opt_state = optimizer.init(model_params)
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
        ckpt_dir = os.path.join("checkpoints", f"bao_{now}")
        rng_key = jax.random.PRNGKey(config.seed)
        iteration = 0
        frames = 0
        hours = 0.0
        grad_steps = 0

    os.makedirs(ckpt_dir, exist_ok=True)
    log: dict = {"iteration": iteration, "hours": hours, "frames": frames}

    while True:
        if iteration % config.eval_interval == 0:
            rng_key, subkey = jax.random.split(rng_key)
            eval_keys = jax.random.split(subkey, n_devices)
            rep_model = jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * n_devices), model
            )
            R, eval_logits, eval_terminated = evaluate(eval_keys, rep_model)
            # Merge device and batch dims back into flat batch
            # (n_devices, batch) → (n_devices*batch,)
            R = R.reshape(-1)
            # (n_devices, 10, batch, actions) → (10, n_devices*batch, actions)
            eval_logits = eval_logits.transpose(1, 0, 2, 3).reshape(
                10, -1, eval_logits.shape[-1]
            )
            # (n_devices, steps, batch) → (steps, n_devices*batch)
            eval_terminated = eval_terminated.transpose(1, 0, 2).reshape(
                config.max_num_steps, -1
            )
            eval_probs_np = jax.device_get(jax.nn.softmax(eval_logits, axis=-1))
            eval_terminated_np = jax.device_get(eval_terminated)
            first_term = np.argmax(eval_terminated_np, axis=0)
            ever_term = eval_terminated_np.any(axis=0)
            eval_game_lengths = np.where(
                ever_term, first_term + 1, config.max_num_steps
            )
            log.update(
                {
                    "eval/vs_random/avg_R": R.mean().item(),
                    "eval/vs_random/win_rate": ((R > 0).sum() / R.size).item(),
                    "eval/vs_random/draw_rate": ((R == 0).sum() / R.size).item(),
                    "eval/vs_random/lose_rate": ((R < 0).sum() / R.size).item(),
                    "eval/game_length/avg": float(eval_game_lengths.mean()),
                    "eval/game_length/min": int(eval_game_lengths.min()),
                    "eval/game_length/max": int(eval_game_lengths.max()),
                }
            )
            if config.use_wandb:
                import wandb

                mean_probs = eval_probs_np[0].mean(axis=0)
                table = wandb.Table(
                    data=[[i, float(p)] for i, p in enumerate(mean_probs)],
                    columns=["move_index", "probability"],
                )
                log["eval/probs/pos_0"] = wandb.plot.bar(
                    table, "move_index", "probability", title="Eval probs at position 0"
                )

                R_np = jax.device_get(R)
                win_lengths = eval_game_lengths[R_np > 0]
                loss_lengths = eval_game_lengths[R_np < 0]
                max_len = config.max_num_steps
                win_counts = np.bincount(win_lengths, minlength=max_len + 1)
                loss_counts = np.bincount(loss_lengths, minlength=max_len + 1)
                win_table = wandb.Table(
                    data=[[int(l), int(win_counts[l])] for l in range(max_len + 1)],
                    columns=["move_count", "wins"],
                )
                loss_table = wandb.Table(
                    data=[[int(l), int(loss_counts[l])] for l in range(max_len + 1)],
                    columns=["move_count", "losses"],
                )
                log["eval/wins_per_move_count"] = wandb.plot.bar(
                    win_table, "move_count", "wins", title="Wins per move count"
                )
                log["eval/losses_per_move_count"] = wandb.plot.bar(
                    loss_table, "move_count", "losses", title="Losses per move count"
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
                        "grad_steps": grad_steps,
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
        selfplay_keys = jax.random.split(subkey, n_devices)
        rep_model = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_devices), model)
        data: SelfplayOutput = selfplay(rep_model, selfplay_keys)
        samples: Sample = compute_loss_input(data)

        # Game length stats — pmap adds leading device dim: (n_devices, max_num_steps, batch)
        # Merge device+batch → (max_num_steps, n_devices*batch)
        terminated_np = jax.device_get(data.terminated)
        terminated_np = terminated_np.transpose(1, 0, 2).reshape(
            config.max_num_steps, -1
        )
        first_term = np.argmax(terminated_np, axis=0)
        ever_term = terminated_np.any(axis=0)
        game_lengths = np.where(ever_term, first_term + 1, config.max_num_steps)

        # Flatten (n_devices, max_num_steps, batch, ...) -> N, shuffle, minibatch
        samples = jax.device_get(samples)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[3:])), samples
        )

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
        per_device_batch = config.training_batch_size // n_devices
        rep_model = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_devices), model)
        rep_opt_state = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * n_devices), opt_state
        )
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            if num_updates == 1:
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape(
                        (n_devices, x.shape[0] // n_devices, *x.shape[1:])
                    ),
                    minibatches,
                )
            else:
                batch: Sample = jax.tree_util.tree_map(
                    lambda x: x[i].reshape((n_devices, per_device_batch, *x.shape[2:])),
                    minibatches,
                )
            rep_model, rep_opt_state, pl, vl = train_step(
                rep_model, rep_opt_state, batch
            )
            policy_losses.append(pl.mean().item())
            value_losses.append(vl.mean().item())
        # Unreplicate: all devices hold the same parameters so take the first
        model = jax.tree_util.tree_map(lambda x: x[0], rep_model)
        opt_state = jax.tree_util.tree_map(lambda x: x[0], rep_opt_state)
        grad_steps += num_updates

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": sum(policy_losses) / len(policy_losses),
                "train/value_loss": sum(value_losses) / len(value_losses),
                "train/learning_rate": float(lr_schedule(grad_steps)),
                "train/game_length/avg": float(game_lengths.mean()),
                "train/game_length/min": int(game_lengths.min()),
                "train/game_length/max": int(game_lengths.max()),
                "hours": hours,
                "frames": frames,
            }
        )
