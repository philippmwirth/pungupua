# ENV — Pgx wrapper design

This document specifies the [pgx](https://github.com/sotetsuk/pgx) `State` /
`Env` wrappers around the internal `bao.Game` / `bao.GameState`.  The wrappers
adapt our game-specific representation to pgx's canonical interface so that
`Bao` can be dropped into pgx-aware code (training loops, MCTS, evaluation
utilities) the same way `pgx.chess.Chess` is.

The split mirrors how `pgx.chess` is organized: the *game logic* lives in
`bao/` (board, sowing, rules, game) and the *pgx adapter* — `State` plus
`Bao(core.Env)` — sits on top.  See [BAO.md](BAO.md) for the underlying
representation.

---

## File layout

```
src/bao/
├── board.py          # (existing) board geometry, sow path, observation tensor
├── sowing.py         # (existing) sow + simulate kernels
├── rules.py          # (existing) namua/mtaji step + legal action mask
├── state.py          # (existing) GameState NamedTuple + constants
├── game.py           # (existing) Game class — init/step/observe/...
├── env.py            # NEW: pgx State + Bao(core.Env)
└── __init__.py       # re-export Game, GameState, State, Bao
```

`env.py` is the **only** new module.  No game logic moves — we just adapt.

---

## Two state types — why both?

| Layer | Type | Purpose |
|---|---|---|
| Internal | `bao.GameState` (NamedTuple) | Authoritative game state.  Compact, JAX-friendly, defined in [state.py](src/bao/state.py). |
| Pgx | `bao.env.State` (`@pgx._src.struct.dataclass`) | What pgx hands to algorithms.  Carries the pgx-required fields (`current_player`, `rewards`, `terminated`, …) plus `_x: GameState`. |

`GameState` keeps the game logic decoupled from pgx so the same code can be
used standalone (e.g. in tests, in `play.py`, or from non-pgx training code).

`GameState` is a `NamedTuple`, which JAX already treats as a pytree — pgx's
`@dataclass` wraps it transparently.  No conversion needed.

---

## `State`

```python
import jax.numpy as jnp
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array

from bao.game import Game
from bao.state import GameState, NUM_ACTIONS

# Computed once at import time: the legal mask for the canonical opening.
_INIT_GAME_STATE = GameState()  # default-constructed -> INIT_BOARD, both stocks 22, etc.
_GAME = Game()
INIT_LEGAL_ACTION_MASK = _GAME.legal_action_mask(_INIT_GAME_STATE)


@dataclass
class State(core.State):
    current_player:    Array = jnp.int32(0)
    rewards:           Array = jnp.zeros(2, jnp.float32)
    terminated:        Array = jnp.bool_(False)
    truncated:         Array = jnp.bool_(False)   # bao has no time limit
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK            # (34,) bool
    observation:       Array = jnp.zeros((4, 8, 67), jnp.float32)
    _step_count:       Array = jnp.int32(0)
    _player_order:     Array = jnp.int32([0, 1])  # randomized in _init
    _x:                GameState = GameState()    # the internal game state

    @property
    def env_id(self) -> core.EnvId:
        return "bao"
```

### Field-by-field

- **`current_player`** — pgx player id (0 or 1) of the player to move.  This
  is *not* the same as `_x.current_player` (the internal id) when
  `_player_order = [1, 0]`.  Derived as `_player_order[_x.current_player]`.
- **`rewards`** — `(2,) float32`, indexed by pgx player id.  Comes from
  `Game.rewards(x)[_player_order]` (see "Player ordering" below).  All zeros
  while the game is ongoing; `+1 / -1` at terminal.
- **`terminated`** — `Game.is_terminal(_x)`.
- **`truncated`** — always `False`; bao has no time-limit truncation.
- **`legal_action_mask`** — `(34,) bool`, computed by
  `Game.legal_action_mask(_x)`.  When `_x.nyumba_pending` it has exactly
  `{32, 33}` set; otherwise it follows the namua/mtaji rules.
- **`observation`** — `(4, 8, 67) float32`.  Channels 0–64 are one-hot seed
  counts (cap = 64); ch 65 = current-player nyumba_active; ch 66 = opponent
  nyumba_active.  The field's default is zeros; pgx fills it in via
  `Env._observe`.
- **`_step_count`** — managed by pgx (incremented automatically).
- **`_player_order`** — `(2,) int32`.  Permutation mapping pgx ↔ internal ids;
  randomized in `_init`.  See below.
- **`_x`** — the internal `GameState`.

---

## `Bao` env

```python
class Bao(core.Env):
    def __init__(self):
        super().__init__()
        self.game = Game()

    @property
    def id(self) -> core.EnvId:       return "bao"
    @property
    def version(self) -> str:         return "v0"
    @property
    def num_players(self) -> int:     return 2

    def _init(self, key: PRNGKey) -> State:
        x = self.game.init()
        # 50/50 which pgx id starts. _player_order[pgx_id] = internal_id.
        _player_order = jnp.array([[0, 1], [1, 0]])[
            jax.random.bernoulli(key).astype(jnp.int32)
        ]
        return State(
            current_player=_player_order[x.current_player],
            legal_action_mask=self.game.legal_action_mask(x),
            _player_order=_player_order,
            _x=x,
        )

    def _step(self, state: core.State, action: Array, key) -> State:
        del key  # bao step is deterministic
        assert isinstance(state, State)
        x = self.game.step(state._x, action)
        return state.replace(
            _x=x,
            legal_action_mask=self.game.legal_action_mask(x),
            terminated=self.game.is_terminal(x),
            rewards=self.game.rewards(x)[state._player_order],
            current_player=state._player_order[x.current_player],
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        # pgx player_id -> internal color
        color = state._player_order[player_id]
        return self.game.observe(state._x, color)
```

---

## Implementation notes

### Player ordering

`_player_order` is a length-2 permutation.  Convention:

```
_player_order[pgx_id] = internal_id
```

So `_player_order[0]` is the *internal* id (0 or 1) corresponding to pgx
player 0, and vice versa.  Both `[0, 1]` and `[1, 0]` are valid; we pick one
uniformly at `_init`.

This gives us:

- `current_player = _player_order[x.current_player]`
  ("the pgx id of whoever's turn it is internally")
- `rewards_pgx = rewards_internal[_player_order]`
  (fancy indexing: `rewards_pgx[i] = rewards_internal[_player_order[i]]`)
- `_observe(state, pgx_id)` reads its color from `_player_order[pgx_id]` and
  delegates straight to `Game.observe(_x, color)` — the existing observe
  already does the board flip when `color != _x.current_player`.

### Why no `_flip` here?

`pgx.chess._observe` calls a `_flip(state._x)` helper to reorient the board
for the opposing perspective and then passes that to `game.observe`.  Bao
doesn't need that step: `Game.observe(state, color)` already handles
"render from `color`'s perspective" internally via `make_observation` in
[board.py](src/bao/board.py).  So our `_observe` is one line.

### Legal action mask is part of `State` (not recomputed on demand)

pgx algorithms expect `state.legal_action_mask` to be cheap.  We compute it
once per `_init` / `_step` and stash it on `State`, mirroring how chess does
it.  The cost is one extra `Game.legal_action_mask` call per step — already
JAX-friendly (jit/vmap-compatible).

### Nyumba pending is invisible to pgx

When the player must choose STOP (32) or CONTINUE (33), the game is
mid-turn from the rules' point of view, but from pgx's point of view it is
just another decision step with `legal_action_mask = {32, 33}`.  Nothing
special is needed in the wrapper.

`current_player` is unchanged across the stop/continue decision because
`Game.step` does not flip while `nyumba_pending`.  pgx sees two consecutive
calls with the same `current_player` — fine.

### Stage transitions are also invisible

Namua → mtaji is handled in `Game.step` (`stage` field flips when both
stocks reach 0).  pgx doesn't care.

### Action space size

`NUM_ACTIONS = 34` is fixed; `legal_action_mask` shape is `(34,)` always.
Re-export `NUM_ACTIONS` from `bao/__init__.py` for downstream code.

### Truncation

Bao has no maximum game length in the rules ("theoretically endless cycles
are possible").  We expose `truncated = False` always.  Callers that need a
move cap should wrap with `pgx.experimental.auto_reset` or apply their own
truncation policy.

### `INIT_LEGAL_ACTION_MASK`

Computed once at module import using `Game()` on the canonical opening
(`board = INIT_BOARD`, stocks `[22, 22]`, stage 0, both nyumbas active, no
pending).  Used as the dataclass default for `legal_action_mask` — pgx
relies on this being a compile-time constant when constructing zero-valued
template states.

The expected init mask has bits set at `{8, 9, 10, 11, 12, 13}` — both
directions on cols 4, 5, and 6 (the three occupied front holes; no capture
is available from the symmetric start).

---

## `__init__.py` exports

```python
# src/bao/__init__.py
from bao.game import Game
from bao.state import GameState, NUM_ACTIONS, NYUMBA_STOP, NYUMBA_CONTINUE
from bao.env import State, Bao, INIT_LEGAL_ACTION_MASK

__all__ = [
    "Game", "GameState",
    "State", "Bao",
    "NUM_ACTIONS", "NYUMBA_STOP", "NYUMBA_CONTINUE",
    "INIT_LEGAL_ACTION_MASK",
]
```

---

## Example usage

```python
import jax
from bao import Bao

env = Bao()
init_fn = jax.jit(env.init)
step_fn = jax.jit(env.step)

key = jax.random.PRNGKey(0)
state = init_fn(key)

while not state.terminated:
    legal = state.legal_action_mask
    action = jax.random.choice(key, jax.numpy.where(legal)[0])
    state = step_fn(state, action, key)

print("rewards:", state.rewards)         # [+1, -1] or [-1, +1]
print("winner:", int(jax.numpy.argmax(state.rewards)))
```

For batched self-play, `jax.vmap(env.init)` / `jax.vmap(env.step)` work
without further changes — every field of `State` is a JAX array.

---

## Open questions / future work

- **Auto-reset**: pgx's training loops typically use `auto_reset` to restart
  a single env after `terminated`.  We don't need to do anything here; it
  works out-of-the-box because `State` is a pgx dataclass.
- **Cycle detection / draws**: the rules say cycles are theoretically
  possible.  If this becomes a problem in self-play, add a repetition count
  to `_x` and have `Game.is_terminal` honor it.  Outside the scope of the
  pgx wrapper.
- **Observation history**: chess stacks 8 historical positions into its
  119-channel observation.  Bao's 67-channel observation is single-frame.
  If MCTS plays poorly without history, extending the wrapper to stack
  `_x.board` over the last N steps is straightforward (`State._history: Array`,
  updated in `_step`).
