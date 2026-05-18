"""Pgx-compatible State and Env wrappers around the internal bao.Game.

See ENV.md for the design rationale.  All game logic lives in `game.py` /
`rules.py` / `sowing.py`; this module only adapts to pgx's interface.
"""

import jax
import jax.numpy as jnp
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

from .game import Game
from .state import GameState

# Pre-compute the legal mask for the canonical opening once at import time so
# it can be the compile-time default on `State.legal_action_mask`.
_GAME = Game()
_INIT_GAME_STATE = _GAME.init()
INIT_LEGAL_ACTION_MASK = _GAME.legal_action_mask(_INIT_GAME_STATE)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.zeros(2, jnp.float32)
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)  # bao has no time limit
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK  # (34,) bool
    observation: Array = jnp.zeros((4, 8, 67), jnp.float32)
    _step_count: Array = jnp.int32(0)
    # Permutation mapping pgx player id -> internal player id, randomized
    # uniformly per episode in `Bao._init`.
    _player_order: Array = jnp.int32([0, 1])
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "bao"  # type: ignore[return-value]


class Bao(core.Env):
    """Bao la Kiswahili as a pgx Env."""

    def __init__(self) -> None:
        super().__init__()
        self.game = Game()

    @property
    def id(self) -> core.EnvId:
        return "bao"  # type: ignore[return-value]

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

    # ------------------------------------------------------------------ pgx hooks

    def _init(self, key: PRNGKey) -> State:
        x = self.game.init()
        # 50/50 which pgx id goes first.  `_player_order[pgx_id] = internal_id`.
        player_order = jnp.array([[0, 1], [1, 0]], jnp.int32)[
            jax.random.bernoulli(key).astype(jnp.int32)
        ]
        return State(  # type: ignore[call-arg]
            current_player=player_order[x.current_player],
            legal_action_mask=self.game.legal_action_mask(x),
            _player_order=player_order,
            _x=x,
        )

    def _step(self, state: core.State, action: Array, key) -> State:
        del key  # bao step is deterministic
        assert isinstance(state, State)
        x = self.game.step(state._x, action)
        return state.replace(  # type: ignore[attr-defined]
            _x=x,
            legal_action_mask=self.game.legal_action_mask(x),
            terminated=self.game.is_terminal(x),
            rewards=self.game.rewards(x)[state._player_order],
            current_player=state._player_order[x.current_player],
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        color = state._player_order[player_id]
        return self.game.observe(state._x, color)
