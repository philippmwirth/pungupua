"""Namua/mtaji move execution and legal action mask."""

import jax
import jax.numpy as jnp
from jax import Array

from .board import kichwa_col, is_kichwa_or_kimbi, forced_direction, next_pos
from .sowing import sow, simulate_sow_ends_in_capture
from .state import NYUMBA_COL, OPP_NYUMBA_COL, NUM_ACTIONS, NYUMBA_STOP, NYUMBA_CONTINUE


# ---------------------------------------------------------------------------
# Namua step
# ---------------------------------------------------------------------------


def namua_step(
    board: Array, stock: Array, nyumba_active: Array, col: Array, direction: Array
):
    """Execute one namua action.

    Returns (new_board, new_nyumba_active, new_stock, nyumba_pending,
             pending_direction, nyumba_emptied).
    """
    # Determine if a capture is possible at this col
    can_capture = (board[0, col] > 0) & (board[2, col] > 0)

    def capturing(args):
        b, na = args
        # Place stock seed
        b = b.at[0, col].add(jnp.int16(1))
        # Take opponent seeds
        captured = b[2, col].astype(jnp.int16)
        b = b.at[2, col].set(jnp.int16(0))
        # Determine sow direction: forced for kichwa/kimbi, else player choice
        eff_dir = jnp.where(is_kichwa_or_kimbi(col), forced_direction(col), direction)
        kc = kichwa_col(eff_dir)
        b, ny_pend, pend_dir, ny_emp = sow(
            b,
            captured,
            jnp.int32(0),
            kc,
            eff_dir,
            allow_capture=True,
            nyumba_active=na[0],
        )
        return b, na, ny_pend, pend_dir, ny_emp

    def non_capturing(args):
        b, na = args
        # Special case: nyumba is the only occupied front hole and has >= 6 seeds.
        # Place stock seed into nyumba, remove 2, sow those 2 from the next hole.
        nyumba_only = (
            (jnp.arange(8, dtype=jnp.int32) == NYUMBA_COL) | (b[0] == 0)
        ).all() & (b[0, NYUMBA_COL] > 0)
        nyumba_special = nyumba_only & (b[0, NYUMBA_COL] >= jnp.int16(6))

        def nyumba_two_seed(inner):
            b2, na2 = inner
            b2 = b2.at[0, NYUMBA_COL].add(jnp.int16(1))  # place stock seed
            b2 = b2.at[0, NYUMBA_COL].add(jnp.int16(-2))  # remove two seeds
            nr, nc = next_pos(jnp.int32(0), jnp.int32(NYUMBA_COL), direction)
            b2, ny_pend, pend_dir, ny_emp = sow(
                b2,
                jnp.int16(2),
                nr,
                nc,
                direction,
                allow_capture=False,
                nyumba_active=na2[0],
            )
            # The 2-seed move never empties the nyumba (>=6 seeds, removes 2).
            return b2, na2, ny_pend, pend_dir, ny_emp

        def normal_takasa(inner):
            b2, na2 = inner
            # Place stock seed, pick up ALL seeds from this hole
            b2 = b2.at[0, col].add(jnp.int16(1))
            seeds = b2[0, col].astype(jnp.int16)
            b2 = b2.at[0, col].set(jnp.int16(0))
            nr, nc = next_pos(jnp.int32(0), col, direction)
            # If the source was the nyumba we just emptied it; treat it as an
            # ordinary hole for the rest of this move (no pending on wraparound).
            src_emp = col == jnp.int32(NYUMBA_COL)
            eff_active = na2[0] & (~src_emp)
            b2, ny_pend, pend_dir, sow_emp = sow(
                b2,
                seeds,
                nr,
                nc,
                direction,
                allow_capture=False,
                nyumba_active=eff_active,
            )
            return b2, na2, ny_pend, pend_dir, sow_emp | src_emp

        return jax.lax.cond(nyumba_special, nyumba_two_seed, normal_takasa, (b, na))

    new_board, new_na, nyumba_pending, pending_dir, nyumba_emptied = jax.lax.cond(
        can_capture, capturing, non_capturing, (board, nyumba_active)
    )
    new_stock = stock.at[0].add(jnp.int16(-1))
    return new_board, new_na, new_stock, nyumba_pending, pending_dir, nyumba_emptied


# ---------------------------------------------------------------------------
# Mtaji step
# ---------------------------------------------------------------------------


def mtaji_step(
    board: Array, nyumba_active: Array, row: Array, col: Array, direction: Array
):
    """Execute one mtaji action.

    Returns (new_board, new_nyumba_active, nyumba_pending, pending_direction,
             nyumba_emptied).
    """
    seeds = board[row, col].astype(jnp.int16)
    new_board = board.at[row, col].set(jnp.int16(0))
    src_emp = (row == jnp.int32(0)) & (col == jnp.int32(NYUMBA_COL))
    eff_active = nyumba_active[0] & (~src_emp)
    nr, nc = next_pos(row, col, direction)
    new_board, nyumba_pending, pending_dir, sow_emp = sow(
        new_board,
        seeds,
        nr,
        nc,
        direction,
        allow_capture=True,
        nyumba_active=eff_active,
    )
    return new_board, nyumba_active, nyumba_pending, pending_dir, sow_emp | src_emp


# ---------------------------------------------------------------------------
# Nyumba continuation
# ---------------------------------------------------------------------------


def nyumba_continue_step(board: Array, nyumba_active: Array):
    """Pick up nyumba seeds and sow from the next hole (direction=right doesn't
    matter here since we need the direction from the pending state).

    NOTE: direction is embedded in the caller; this function is called from
    game.py which passes the stored direction.  We expose a version that takes
    direction explicitly.
    """
    pass  # implemented inline in game.py


# ---------------------------------------------------------------------------
# Nyumba deactivation
# ---------------------------------------------------------------------------


def deactivate_nyumba_if_sowed(
    board: Array,
    nyumba_active: Array,
    prev_board: Array,
    cur_nyumba_emptied: Array = None,
) -> Array:
    """Mark nyumbas inactive when their seeds were sown out during this move.

    Current player's nyumba is deactivated when its seeds were picked up at
    any point during the move (RULES: "once the nyumba's seeds have been sown
    out, it becomes an ordinary hole").  This includes the case where the
    nyumba was emptied mid-sow but later refilled by the same chain — the
    `cur_nyumba_emptied` flag from the sow captures that.  We fall back to
    the prev/final comparison when the flag is not provided (e.g. callers
    that do not yet thread it through).

    Opponent's nyumba: deactivated when board[2, OPP_NYUMBA_COL] drops to 0
    (seeds were captured) — captures cannot be undone within the same move.
    """
    final_empty = (prev_board[0, NYUMBA_COL] > 0) & (board[0, NYUMBA_COL] == 0)
    if cur_nyumba_emptied is None:
        cur_nyumba_emptied = jnp.bool_(False)
    cur_emptied = nyumba_active[0] & (final_empty | cur_nyumba_emptied)
    opp_emptied = (
        nyumba_active[1]
        & (prev_board[2, OPP_NYUMBA_COL] > 0)
        & (board[2, OPP_NYUMBA_COL] == 0)
    )
    return jnp.array([nyumba_active[0] & ~cur_emptied, nyumba_active[1] & ~opp_emptied])


# ---------------------------------------------------------------------------
# Legal action mask
# ---------------------------------------------------------------------------


def capture_possible_namua(board: Array, stock: Array) -> Array:
    """True if any front-row col has seeds on both sides and stock > 0."""
    has_stock = stock[0] > 0
    both = (board[0] > 0) & (board[2] > 0)
    return has_stock & both.any()


_capture_possible_namua = capture_possible_namua  # internal alias


def _namua_mask(board_stock_etc) -> Array:
    board, stock, nyumba_active, opp_nyumba_active = board_stock_etc
    has_stock = stock[0] > 0
    can_cap = _capture_possible_namua(board, stock)

    # Nyumba-only special case (RULES §Takasa with Only the Nyumba Remaining):
    # applies when nyumba is the sole occupied front hole and has >= 6 seeds.
    nyumba_only = (
        (jnp.arange(8, dtype=jnp.int32) == NYUMBA_COL) | (board[0] == 0)
    ).all() & (board[0, NYUMBA_COL] > 0)
    nyumba_special = (
        has_stock & nyumba_only & ~can_cap & (board[0, NYUMBA_COL] >= jnp.int16(6))
    )

    # Build mask over front-row cols (actions 0-15)
    def col_mask(c):
        eligible_cap = has_stock & (board[0, c] > 0) & (board[2, c] > 0)
        eligible_tak = has_stock & (board[0, c] > 0)
        eligible_nyumba_sp = nyumba_special & (c == jnp.int32(NYUMBA_COL))

        # Direction is forced for kichwa/kimbi ONLY when capturing (RULES:
        # "If you capture from a kichwa or kimbi hole, you must sow from the
        # same side").  Plain takasa from a kichwa/kimbi hole is unconstrained.
        is_special = is_kichwa_or_kimbi(c)
        fdir = forced_direction(c)
        cap_left_ok = (~is_special) | (fdir == 0)
        cap_right_ok = (~is_special) | (fdir == 1)
        left_ok = jnp.where(can_cap, cap_left_ok, jnp.bool_(True))
        right_ok = jnp.where(can_cap, cap_right_ok, jnp.bool_(True))

        eligible = jnp.where(
            can_cap,
            eligible_cap,
            jnp.where(nyumba_special, eligible_nyumba_sp, eligible_tak),
        )
        a_left = (c * 2 + 0).astype(jnp.int32)
        a_right = (c * 2 + 1).astype(jnp.int32)
        return eligible & left_ok, a_left, eligible & right_ok, a_right

    mask = jnp.zeros(NUM_ACTIONS, jnp.bool_)
    for ci in range(8):
        c = jnp.int32(ci)
        el, al, er, ar = col_mask(c)
        mask = mask.at[al].set(el)
        mask = mask.at[ar].set(er)
    return mask


# Vectorised mtaji capture check
def _mtaji_capture_mask(board: Array, nyumba_active: Array) -> Array:
    """(32,) bool: which (row, col, direction) actions produce a capture."""
    cur_active = nyumba_active[0]

    def check(action):
        row = jnp.where(action < 16, jnp.int32(0), jnp.int32(1))
        rem = jnp.where(action < 16, action, action - jnp.int32(16))
        col = rem // 2
        direction = rem % 2
        singleton = board[row, col] <= 1
        return (~singleton) & simulate_sow_ends_in_capture(
            board,
            row,
            col,
            direction,
            nyumba_active=cur_active,
        )

    actions = jnp.arange(32, dtype=jnp.int32)
    return jax.vmap(check)(actions)


def _mtaji_mask(board_stock_etc) -> Array:
    board, stock, nyumba_active, opp_nyumba_active = board_stock_etc

    cap_mask32 = _mtaji_capture_mask(board, nyumba_active)
    any_capture = cap_mask32.any()

    # Mtaji-moja: active when the opponent (board[2]) has exactly one non-singleton
    # hole that is directly paired with one of our front-row holes (board[0,c]>0).
    # Singletons in board[2] are excluded because we cannot sow singletons in mtaji,
    # so they are not real capture threats.
    paired_nonsing = (board[0] > 0) & (board[2] >= jnp.int16(2))
    mtaji_moja_active = paired_nonsing.sum() == 1

    def capture_actions(_):
        # Direction constraint: kichwa/kimbi forces direction for front-row only
        def dir_ok(action):
            row = jnp.where(action < 16, jnp.int32(0), jnp.int32(1))
            rem = jnp.where(action < 16, action, action - jnp.int32(16))
            col = rem // 2
            direction = rem % 2
            is_special = is_kichwa_or_kimbi(col) & (row == 0)
            fdir = forced_direction(col)
            return (~is_special) | (fdir == direction)

        actions = jnp.arange(32, dtype=jnp.int32)
        dok = jax.vmap(dir_ok)(actions)
        return (cap_mask32 & dok).astype(jnp.bool_)

    def takasa_actions(_):
        # Front row preferred; back row is a fallback when no front-row
        # non-singletons exist (RULES §Takasa in Mtaji).
        # No kichwa/kimbi direction constraint — that applies only when
        # capturing FROM a kichwa/kimbi (RULES §Kichwa and Kimbi).
        def front_eligible(action):
            row = jnp.where(action < 16, jnp.int32(0), jnp.int32(1))
            rem = jnp.where(action < 16, action, action - jnp.int32(16))
            col = rem // 2
            singleton = board[row, col] <= 1
            is_front = row == 0
            only_opp_col = jnp.argmax(paired_nonsing)
            is_moja_source = mtaji_moja_active & is_front & (col == only_opp_col)
            return is_front & (~singleton) & (~is_moja_source)

        def back_eligible(action):
            row = jnp.where(action < 16, jnp.int32(0), jnp.int32(1))
            rem = jnp.where(action < 16, action, action - jnp.int32(16))
            col = rem // 2
            singleton = board[row, col] <= 1
            return (row == 1) & (~singleton)

        actions = jnp.arange(32, dtype=jnp.int32)
        front_mask = jax.vmap(front_eligible)(actions).astype(jnp.bool_)
        back_mask = jax.vmap(back_eligible)(actions).astype(jnp.bool_)
        return jnp.where(front_mask.any(), front_mask, back_mask)

    mask32 = jax.lax.cond(any_capture, capture_actions, takasa_actions, None)
    mask = jnp.zeros(NUM_ACTIONS, jnp.bool_)
    mask = mask.at[:32].set(mask32)
    return mask


# Wrap for external call with positional args matching game.py usage
def compute_legal_action_mask(board, stock, stage, nyumba_active, nyumba_pending):
    opp_nyumba_active = nyumba_active[1]

    def nyumba_mask(_):
        m = jnp.zeros(NUM_ACTIONS, jnp.bool_)
        return m.at[NYUMBA_STOP].set(True).at[NYUMBA_CONTINUE].set(True)

    def normal_mask(_):
        packed = (board, stock, nyumba_active, opp_nyumba_active)
        return jax.lax.cond(stage == 0, _namua_mask, _mtaji_mask, packed)

    return jax.lax.cond(nyumba_pending, nyumba_mask, normal_mask, None)
