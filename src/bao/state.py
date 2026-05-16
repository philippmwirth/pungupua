from typing import NamedTuple
import jax.numpy as jnp
from jax import Array


class GameState(NamedTuple):
    # (4, 8) int16 seed counts stored from current player's perspective:
    #   row 0 = current player's front row (col 0 = their leftmost)
    #   row 1 = current player's back row
    #   row 2 = opponent's front row (col i directly opposes board[0, i])
    #   row 3 = opponent's back row
    board: Array = jnp.zeros((4, 8), jnp.int16)
    # [current_player_stock, opponent_stock]
    stock: Array = jnp.array([22, 22], jnp.int16)
    # canonical player id
    current_player: Array = jnp.int32(0)
    # 0 = namua, 1 = mtaji
    stage: Array = jnp.int32(0)
    # -1 = no winner
    winner: Array = jnp.int32(-1)
    # [current_player_nyumba_active, opponent_nyumba_active]
    nyumba_active: Array = jnp.array([True, True])
    # True when player must choose action 32 (stop) or 33 (continue)
    nyumba_pending: Array = jnp.bool_(False)
    # Direction that was active when nyumba_pending was set (0=left, 1=right)
    pending_direction: Array = jnp.int32(1)


# Starting board from current player's perspective
INIT_BOARD = jnp.array([
    [0, 0, 0, 0, 6, 2, 2, 0],  # current player front: nyumba=6 at col4
    [0, 0, 0, 0, 0, 0, 0, 0],  # current player back
    [0, 2, 2, 6, 0, 0, 0, 0],  # opponent front: nyumba=6 at col3 (their col4 mirrored)
    [0, 0, 0, 0, 0, 0, 0, 0],  # opponent back
], jnp.int16)

NYUMBA_COL = 4
LEFT_KICHWA = 0
LEFT_KIMBI = 1
RIGHT_KIMBI = 6
RIGHT_KICHWA = 7

# Action encoding:
#   0-15:  front row, action = col*2 + direction  (0=left, 1=right)
#   16-31: back row,  action = 16 + col*2 + direction
#   32:    nyumba stop
#   33:    nyumba continue
NUM_ACTIONS = 34
NYUMBA_STOP = 32
NYUMBA_CONTINUE = 33
