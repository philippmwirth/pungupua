"""Board-level helpers: turn flip, sow-path indexing, observation encoding."""

import jax
import jax.numpy as jnp
from jax import Array


def flip_board(board: Array, nyumba_active: Array, stock: Array):
    """Flip board so the next player sees themselves at rows 0-1."""
    new_board = jnp.stack(
        [
            board[2, ::-1],
            board[3, ::-1],
            board[0, ::-1],
            board[1, ::-1],
        ]
    )
    new_nyumba = jnp.array([nyumba_active[1], nyumba_active[0]])
    new_stock = jnp.array([stock[1], stock[0]], jnp.int16)
    return new_board, new_nyumba, new_stock


def action_to_row_col_dir(action: Array):
    """Decode action 0-31 into (row, col, direction). Actions 32-33 are not decoded here."""
    row = jnp.where(action < 16, jnp.int32(0), jnp.int32(1))
    remainder = jnp.where(action < 16, action, action - jnp.int32(16))
    col = remainder // 2
    direction = remainder % 2  # 0=left, 1=right
    return row, col, direction


def next_pos(row: Array, col: Array, direction: Array):
    """Advance one step along the sow path within the current player's half (rows 0-1).

    Sowing right: front[0->7] then back[7->0], wrapping at the right end.
    Sowing left:  front[7->0] then back[0->7], wrapping at the left end.
    """
    # Encode (row, col) as position 0-15 on a circular path.
    # Right path: front col 0->7 = pos 0->7, back col 7->0 = pos 8->15
    # Left  path: front col 7->0 = pos 0->7, back col 0->7 = pos 8->15
    pos_right = jnp.where(row == 0, col, jnp.int32(15) - col)
    pos_left = jnp.where(row == 0, jnp.int32(7) - col, jnp.int32(8) + col)

    pos = jnp.where(direction == 1, pos_right, pos_left)
    pos = (pos + 1) % 16

    # Decode position back to (row, col)
    nr_right = jnp.where(pos < 8, jnp.int32(0), jnp.int32(1))
    nc_right = jnp.where(pos < 8, pos, jnp.int32(15) - pos)

    nr_left = jnp.where(pos < 8, jnp.int32(0), jnp.int32(1))
    nc_left = jnp.where(pos < 8, jnp.int32(7) - pos, pos - jnp.int32(8))

    new_row = jnp.where(direction == 1, nr_right, nr_left)
    new_col = jnp.where(direction == 1, nc_right, nc_left)
    return new_row.astype(jnp.int32), new_col.astype(jnp.int32)


def kichwa_col(direction: Array) -> Array:
    """Kichwa column to start sowing from for the given direction.

    Sowing right (direction=1) starts at the LEFT kichwa (col 0).
    Sowing left  (direction=0) starts at the RIGHT kichwa (col 7).
    """
    return jnp.where(direction == 1, jnp.int32(0), jnp.int32(7))


def is_kichwa_or_kimbi(col: Array) -> Array:
    return (col == 0) | (col == 1) | (col == 6) | (col == 7)


def forced_direction(col: Array) -> Array:
    """For kichwa/kimbi cols, return the forced sowing direction.

    Left kichwa/kimbi (cols 0,1): captured seeds enter from left kichwa -> sow right (1).
    Right kichwa/kimbi (cols 6,7): captured seeds enter from right kichwa -> sow left (0).
    """
    return jnp.where(col <= 1, jnp.int32(1), jnp.int32(0))


def make_observation(
    board: Array,
    nyumba_active: Array,
    color: Array,
    current_player: Array,
    stock: Array,
    stage: Array,
    nyumba_pending: Array,
    pending_direction: Array,
    pending_allow_capture: Array,
) -> Array:
    """Return (4, 8, 73) float32 observation from the perspective of `color`.

    Channels (last axis):
      0-64   : one-hot seed counts for each of the 4×8 holes (65 bins, 0-64)
      65     : current-player nyumba active flag
      66     : opponent nyumba active flag
      67     : current-player stock, normalised to [0, 1] by dividing by 22
      68     : opponent stock, normalised to [0, 1] by dividing by 22
      69     : stage (0 = namua, 1 = mtaji)
      70     : nyumba_pending flag
      71     : pending_direction (−1 = left, 0 = no pending, +1 = right)
      72     : pending_allow_capture flag
    """
    is_cur = color == current_player
    obs_board = jax.lax.cond(
        is_cur,
        lambda b: b,
        lambda b: jnp.stack([b[2, ::-1], b[3, ::-1], b[0, ::-1], b[1, ::-1]]),
        board,
    )
    obs_nyumba = jax.lax.cond(
        is_cur,
        lambda n: n,
        lambda n: jnp.array([n[1], n[0]]),
        nyumba_active,
    )
    # Stock is also perspective-dependent: [my_stock, opp_stock].
    obs_stock = jax.lax.cond(
        is_cur,
        lambda s: s,
        lambda s: jnp.array([s[1], s[0]], jnp.int16),
        stock,
    )

    # One-hot encode seed counts: 65 bins for counts 0-64
    one_hot = jax.nn.one_hot(obs_board, 65, dtype=jnp.float32)  # (4, 8, 65)

    def scalar_plane(v):
        return jnp.full((4, 8, 1), v.astype(jnp.float32))

    n0  = scalar_plane(obs_nyumba[0])
    n1  = scalar_plane(obs_nyumba[1])
    s0  = scalar_plane(obs_stock[0].astype(jnp.float32) / 22.0)
    s1  = scalar_plane(obs_stock[1].astype(jnp.float32) / 22.0)
    stg = scalar_plane(stage.astype(jnp.float32))
    np_ = scalar_plane(nyumba_pending.astype(jnp.float32))
    # When no pending decision, direction and allow_capture are meaningless —
    # zero them out so the NN sees a clean signal.
    # Direction is encoded as −1 (left), 0 (none), +1 (right).
    pd  = scalar_plane(((2 * pending_direction - 1) * nyumba_pending).astype(jnp.float32))
    pac = scalar_plane((pending_allow_capture * nyumba_pending).astype(jnp.float32))

    return jnp.concatenate(
        [one_hot, n0, n1, s0, s1, stg, np_, pd, pac], axis=-1
    )  # (4, 8, 73)
