"""Shared bot-loading utilities for eval scripts."""

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import mctx

from bao import NUM_ACTIONS
from train.network import AZNet


def parse_bots_file(path: str) -> list[dict]:
    """Load bots JSONL/JSON file; tolerates Python-style trailing commas."""
    text = Path(path).read_text()
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return json.loads(text)


def make_forward(num_channels: int, num_blocks: int):
    """Return a Haiku-transformed forward function for AZNet."""

    def _fn(x, is_eval: bool = False):
        net = AZNet(
            num_actions=NUM_ACTIONS,
            num_channels=num_channels,
            num_blocks=num_blocks,
        )
        return net(x, is_training=not is_eval, test_local_stats=False)

    return hk.without_apply_rng(hk.transform_with_state(_fn))


def load_model(ckpt_path: str, num_channels: int, num_blocks: int):
    """Load a checkpoint and return (model, forward)."""
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    forward = make_forward(num_channels, num_blocks)
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["model"], forward


def make_recurrent_fn(game, forward):
    """Return an mctx-compatible recurrent_fn for the given game and network."""

    def recurrent_fn(model, _rng_key, action, state):
        model_params, model_state = model
        prev_player = state.current_player

        new_state = jax.vmap(game.step)(state, action)
        obs = jax.vmap(game.observe)(new_state)
        (logits, value), _ = forward.apply(model_params, model_state, obs, is_eval=True)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        mask = jax.vmap(game.legal_action_mask)(new_state)
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)

        reward = jax.vmap(lambda s, p: game.rewards(s)[p])(new_state, prev_player)
        value = jnp.where(new_state.winner >= 0, 0.0, value)
        player_changed = new_state.current_player != prev_player
        discount = jnp.where(
            new_state.winner >= 0, 0.0, jnp.where(player_changed, -1.0, 1.0)
        )
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        ), new_state

    return recurrent_fn


def run_mcts(
    model,
    forward,
    game,
    states,
    rng_key,
    num_simulations: int,
    gumbel_scale: float = 0.0,
):
    """Run Gumbel MuZero MCTS and return the policy output."""
    model_params, model_state = model
    recurrent_fn = make_recurrent_fn(game, forward)

    obs = jax.vmap(game.observe)(states)
    mask = jax.vmap(game.legal_action_mask)(states)
    (logits, value), _ = forward.apply(model_params, model_state, obs, is_eval=True)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=states)

    return mctx.gumbel_muzero_policy(
        params=model,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=gumbel_scale,
    )


@dataclass
class Bot:
    name: str
    icon: str
    description: str
    num_simulations: int
    gumbel_scale: float
    checkpoint_path: str
    model: Any  # (params, state) loaded from checkpoint
    forward: Any  # haiku transformed fn
    _action_fn: Any = field(default=None, repr=False)

    def build_action_fn(self, game) -> None:
        """JIT-compile a batched MCTS action function for tournament play."""
        forward = self.forward
        num_sims = self.num_simulations
        gscale = self.gumbel_scale

        @jax.jit
        def _get_actions(model, states, rng_key):
            return run_mcts(
                model, forward, game, states, rng_key, num_sims, gscale
            ).action

        self._action_fn = _get_actions

    def get_actions(self, states, rng_key) -> jax.Array:
        return self._action_fn(self.model, states, rng_key)

    def search(self, game, state, rng_key) -> Any:
        """Run MCTS on a single (unbatched) state and return the full policy output."""
        batched = jax.tree_util.tree_map(lambda x: x[None], state)
        return run_mcts(
            self.model,
            self.forward,
            game,
            batched,
            rng_key,
            self.num_simulations,
            gumbel_scale=0.0,
        )


def load_bot(d: dict, game=None) -> "Bot":
    """Load a bot from a config dict. Pass game to also JIT-compile get_actions."""
    mc = d["model_config"]
    ec = d["eval_config"]
    model, forward = load_model(
        d["checkpoint_path"], mc["num_channels"], mc["num_blocks"]
    )
    bot = Bot(
        name=d["name"],
        icon=d.get("icon", "🤖"),
        description=d.get("description", ""),
        num_simulations=ec["num_simulations"],
        gumbel_scale=ec["gumbel_scale"],
        checkpoint_path=d["checkpoint_path"],
        model=model,
        forward=forward,
    )
    if game is not None:
        bot.build_action_fn(game)
    return bot
