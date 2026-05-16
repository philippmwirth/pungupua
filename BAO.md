# Bao la Kiswahili — Implementation Design

Implementation of Bao la Kiswahili as a [pgx](https://github.com/sotetsuk/pgx)-compatible game for JAX-based game tree search (e.g. [mctx](https://github.com/google-deepmind/mctx)).

See [RULES.md](RULES.md) for the full game rules.

---

## `GameState`

```python
class GameState(NamedTuple):
    board:          Array  # (4, 8)  int16  — seed counts, current player's perspective
    stock:          Array  # (2,)    int16  — [current_player, opponent]
    current_player: Array  # ()      int32  — canonical id 0 or 1
    stage:          Array  # ()      int32  — 0 = namua, 1 = mtaji
    winner:         Array  # ()      int32  — -1 = none
    nyumba_active:  Array  # (2,)    bool   — [current, opponent]
    nyumba_pending: Array  # ()      bool   — awaiting stop/continue decision
```

---

## Action Space — 34 actions

| Range | Meaning |
|---|---|
| 0–15  | Front row: `action = col * 2 + direction` |
| 16–31 | Back row:  `action = 16 + col * 2 + direction` |
| 32    | Nyumba: **stop** (end turn) |
| 33    | Nyumba: **continue** (sow from next hole) |

`direction`: 0 = left (decreasing col), 1 = right (increasing col).

Actions 32–33 are the **only** legal actions when `nyumba_pending = True`.

---

## Board Convention

The board is always stored from the **current player's perspective**:

```
board[0]  current player's front row   col 0 = their leftmost hole
board[1]  current player's back row
board[2]  opponent's front row         col i is directly across from board[0, i]
board[3]  opponent's back row
```

Fixed landmarks (always true regardless of whose turn it is):

| Landmark      | Index        |
|---|---|
| Left kichwa   | `board[0, 0]` |
| Left kimbi    | `board[0, 1]` |
| Nyumba        | `board[0, 4]` |
| Right kimbi   | `board[0, 6]` |
| Right kichwa  | `board[0, 7]` |

### Starting Position

Both players start symmetrically. From the current player's perspective:

```
board[0] = [0, 0, 0, 0, 6, 2, 2, 0]   # front: nyumba=6 at col4, 2 at col5, 2 at col6
board[1] = [0, 0, 0, 0, 0, 0, 0, 0]   # back: empty
board[2] = [0, 2, 2, 6, 0, 0, 0, 0]   # opponent front (col3 = opponent's nyumba)
board[3] = [0, 0, 0, 0, 0, 0, 0, 0]   # opponent back: empty
stock    = [22, 22]
```

### Turn Flip

At the end of every turn the board is flipped so the new current player always sees their own pieces at rows 0–1:

```python
new_board = jnp.stack([
    old_board[2, ::-1],   # opponent front  → new current front
    old_board[3, ::-1],   # opponent back   → new current back
    old_board[0, ::-1],   # current front   → new opponent front
    old_board[1, ::-1],   # current back    → new opponent back
])
# also swap: stock, nyumba_active
```

---

## Sowing Rules

Verified against diagrams 14–18 of the rules.

### Landing Conditions

| Situation at landing hole | What happens |
|---|---|
| Empty hole | Turn ends |
| Occupied, front row, opposing hole occupied | **Capture**: take opposing seeds; re-enter from the kichwa matching current direction |
| Occupied, opposing empty (or back row) | **Continue**: pick up all seeds here; sow from the *next* hole in same direction |
| Occupied, **nyumba**, opposing empty | Set `nyumba_pending = True`; player chooses action 32 (stop) or 33 (continue) |

### Sow Path

The path wraps around the player's own half only:

- **Right**: `front[0→7]` → `back[7→0]` → repeat
- **Left**: `front[7→0]` → `back[0→7]` → repeat

### Direction Rules

| Situation | Direction |
|---|---|
| Captured from left kichwa or left kimbi | Must sow **left** |
| Captured from right kichwa or right kimbi | Must sow **right** |
| Already sowing in a direction | Continue same direction |
| Free choice | Player picks via action |

### Takasa

- **Namua**: no capture possible → place stock seed in any front-row hole with ≥ 1 seed (making it ≥ 2), pick up all seeds, sow. No captures during the move.
- **Mtaji**: no capture possible → sow any non-singleton front-row hole. Cannot sow the opponent's sole remaining mtaji target (mtaji moja rule).

### Singleton Rule

A hole containing exactly 1 seed may never be sowed in mtaji. In namua takasa the stock seed is added first (making it 2), so those moves are allowed.

### Implementation

Sowing is implemented as a `jax.lax.while_loop` over a carry:

```python
SowCarry = (board, seeds_in_hand, row, col, direction, done, nyumba_pending)
```

Maximum iterations: **256** (safe upper bound for 64 total seeds on the board).

---

## Observation — shape `(4, 8, 67)`

```
obs[:, :, 0:65]   one-hot seed count per hole — jnp.eye(65)[board]
obs[:, :,   65]   nyumba_active[0] broadcast  — current player's nyumba
obs[:, :,   66]   nyumba_active[1] broadcast  — opponent's nyumba
```

One-hot uses 65 bins (counts 0–64, the maximum possible seeds in one hole).

When `color != current_player`, the board is flipped before encoding using the same turn-flip operation.

---

## Win Conditions

Checked after every `step` call:

1. `board[2].sum() == 0` — opponent's entire front row is empty.
2. `legal_action_mask(next_state).any() == False` — opponent has no legal moves.

---

## `Game` API

```python
class Game:
    def init(self) -> GameState
    def step(self, state: GameState, action: Array) -> GameState
    def observe(self, state: GameState, color: Array) -> Array        # (4, 8, 67) float32
    def legal_action_mask(self, state: GameState) -> Array            # (34,) bool
    def is_terminal(self, state: GameState) -> Array                  # () bool
    def rewards(self, state: GameState) -> Array                      # (2,) float32
```

### `step` Logic

```
if nyumba_pending:
    action 32 → end turn (flip board, switch player)
    action 33 → pick up nyumba seeds, sow from next hole (while_loop)
else if stage == namua:
    if capturing:
        place stock seed into board[0, col]
        take all seeds from board[2, col]
        determine direction (kichwa/kimbi rule or player choice)
        sow_loop(captured_seeds, from_kichwa, direction)
    else (takasa):
        place stock seed into board[0, col]
        pick up all seeds from board[0, col]
        sow_loop(seeds, col, direction)        # no captures allowed
else (mtaji):
    pick up seeds from (row, col)
    sow_loop(seeds, row, col, direction)

after sowing:
    if namua: decrement stock[0]
    if both stocks empty: stage → mtaji
    check win conditions → set winner
    if not terminal: flip board, swap stock/nyumba_active, increment current_player
```

### `legal_action_mask` Logic

```
if nyumba_pending:
    return [False * 32, True, True]   # only actions 32 and 33

capture_mask = for each (col, direction) in front row:
                   board[0, col] > 0
                   AND board[2, col] > 0
                   AND (namua → stock[0] > 0)
                   AND direction consistent with kichwa/kimbi rules

if mtaji and capture via back row:
    also check (row=1, col, direction) combos via mini-sow simulation

if capture_mask.any():
    return capture_mask (padded to 34)
else (takasa):
    return non-singleton front-row holes, both directions,
           excluding mtaji-moja target if applicable
```

---

## File Layout

```
kikande-2/
  bao/
    __init__.py
    state.py      # GameState, starting position constants
    board.py      # flip_board, sow-path indexing, one-hot obs
    sowing.py     # while_loop sow kernel + SowCarry type
    rules.py      # namua_step, mtaji_step, legal_action_mask helpers
    game.py       # Game class
  tests/
    test_diagrams.py   # unit tests reproducing diagrams 3–21 from RULES.md
    test_game.py       # seed conservation, no illegal captures
  BAO.md
  RULES.md
```
