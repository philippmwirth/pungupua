"""Namua/mtaji move execution and legal action mask."""
import jax
import jax.numpy as jnp
from jax import Array

from .board import kichwa_col, is_kichwa_or_kimbi, forced_direction, action_to_row_col_dir
from .sowing import sow, simulate_sow_ends_in_capture
from .state import NYUMBA_COL, NUM_ACTIONS, NYUMBA_STOP, NYUMBA_CONTINUE


# ---------------------------------------------------------------------------
# Namua step
# ---------------------------------------------------------------------------

def namua_step(board: Array, stock: Array, nyumba_active: Array,
               col: Array, direction: Array):
    """Execute one namua action.

    Returns (new_board, new_nyumba_active, new_stock, nyumba_pending).
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
        b, ny_pend, pend_dir = sow(b, captured, jnp.int32(0), kc, eff_dir, allow_capture=True)
        return b, na, ny_pend, pend_dir

    def takasa(args):
        b, na = args
        # Place stock seed, pick up ALL seeds from this hole
        b = b.at[0, col].add(jnp.int16(1))
        seeds = b[0, col].astype(jnp.int16)
        b = b.at[0, col].set(jnp.int16(0))
        # Sow — no captures allowed during takasa
        b, ny_pend, pend_dir = sow(b, seeds, jnp.int32(0), col, direction, allow_capture=False)
        return b, na, ny_pend, pend_dir

    new_board, new_na, nyumba_pending, pending_dir = jax.lax.cond(
        can_capture, capturing, takasa, (board, nyumba_active)
    )
    new_stock = stock.at[0].add(jnp.int16(-1))
    return new_board, new_na, new_stock, nyumba_pending, pending_dir


# ---------------------------------------------------------------------------
# Mtaji step
# ---------------------------------------------------------------------------

def mtaji_step(board: Array, nyumba_active: Array,
               row: Array, col: Array, direction: Array):
    """Execute one mtaji action.

    Returns (new_board, new_nyumba_active, nyumba_pending, pending_direction).
    """
    seeds = board[row, col].astype(jnp.int16)
    new_board = board.at[row, col].set(jnp.int16(0))
    new_board, nyumba_pending, pending_dir = sow(new_board, seeds, row, col,
                                                  direction, allow_capture=True)
    return new_board, nyumba_active, nyumba_pending, pending_dir


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

def deactivate_nyumba_if_sowed(board: Array, nyumba_active: Array,
                                prev_board: Array) -> Array:
    """Mark current player's nyumba inactive if its seeds were dispersed."""
    was_active = nyumba_active[0]
    # Nyumba is deactivated once its seeds are sowed (hole becomes 0 from nonzero,
    # or more precisely: it was nonzero before and is now part of sowing history).
    # We use a simple heuristic: if prev nyumba had seeds and was active, and now
    # it still exists as ordinary hole (seeds changed), check if it was sowed.
    # The cleanest signal: if the nyumba col was ever picked up as a sow start,
    # prev_board[0, NYUMBA_COL] > 0 and board[0, NYUMBA_COL] == 0 after a
    # continue-from-nyumba action.  But during normal captures sowing passes
    # through the nyumba.
    # Conservative rule used here: nyumba deactivates only when explicitly sowed
    # via the nyumba_continue action (handled in game.py).  This function handles
    # the case where the nyumba happens to become 0 due to a continue-sow through it.
    nyumba_emptied = was_active & (prev_board[0, NYUMBA_COL] > 0) & (board[0, NYUMBA_COL] == 0)
    new_na0 = nyumba_active[0] & (~nyumba_emptied)
    return nyumba_active.at[0].set(new_na0)


# ---------------------------------------------------------------------------
# Legal action mask
# ---------------------------------------------------------------------------

def legal_action_mask(board: Array, stock: Array, stage: Array,
                      nyumba_active: Array, nyumba_pending: Array,
                      opp_nyumba_active: Array) -> Array:
    """Return (NUM_ACTIONS,) bool mask of legal actions."""

    # --- nyumba decision ---
    def nyumba_mask(_):
        m = jnp.zeros(NUM_ACTIONS, jnp.bool_)
        m = m.at[NYUMBA_STOP].set(True)
        m = m.at[NYUMBA_CONTINUE].set(True)
        return m

    def normal_mask(_):
        return jax.lax.cond(stage == 0, _namua_mask, _mtaji_mask, None)

    return jax.lax.cond(nyumba_pending, nyumba_mask, normal_mask, None)


def _capture_possible_namua(board: Array, stock: Array) -> Array:
    """True if any front-row col has seeds on both sides and stock > 0."""
    has_stock = stock[0] > 0
    both = (board[0] > 0) & (board[2] > 0)
    return has_stock & both.any()


def _namua_mask(board_stock_etc) -> Array:
    board, stock, nyumba_active, opp_nyumba_active = board_stock_etc
    has_stock = stock[0] > 0
    can_cap = _capture_possible_namua(board, stock)

    # Build mask over front-row cols (actions 0-15)
    cols = jnp.arange(8, dtype=jnp.int32)

    def col_mask(c):
        # Base eligibility
        eligible_cap = has_stock & (board[0, c] > 0) & (board[2, c] > 0)
        eligible_tak = has_stock & (board[0, c] > 0)  # singletons ok (stock adds 1)

        # Direction constraints for kichwa/kimbi
        is_special = is_kichwa_or_kimbi(c)
        fdir = forced_direction(c)
        left_ok  = (~is_special) | (fdir == 0)
        right_ok = (~is_special) | (fdir == 1)

        eligible = jnp.where(can_cap, eligible_cap, eligible_tak)
        a_left  = (c * 2 + 0).astype(jnp.int32)
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
def _mtaji_capture_mask(board: Array) -> Array:
    """(32,) bool: which (row, col, direction) actions produce a capture."""
    def check(action):
        row = jnp.where(action < 16, jnp.int32(0), jnp.int32(1))
        rem = jnp.where(action < 16, action, action - jnp.int32(16))
        col = rem // 2
        direction = rem % 2
        singleton = board[row, col] <= 1
        return (~singleton) & simulate_sow_ends_in_capture(board, row, col, direction)

    actions = jnp.arange(32, dtype=jnp.int32)
    return jax.vmap(check)(actions)


def _mtaji_mask(board_stock_etc) -> Array:
    board, stock, nyumba_active, opp_nyumba_active = board_stock_etc

    cap_mask32 = _mtaji_capture_mask(board)
    any_capture = cap_mask32.any()

    # Find the single mtaji-moja col (opponent's only occupied front col),
    # if there is exactly one.
    opp_front_occupied = board[2] > 0
    opp_count = opp_front_occupied.sum()
    mtaji_moja_active = opp_count == 1

    def capture_actions(_):
        # Direction constraint: kichwa/kimbi forces direction
        def dir_ok(action):
            rem = jnp.where(action < 16, action, action - jnp.int32(16))
            col = rem // 2
            direction = rem % 2
            is_special = is_kichwa_or_kimbi(col)
            fdir = forced_direction(col)
            return (~is_special) | (fdir == direction)

        actions = jnp.arange(32, dtype=jnp.int32)
        dok = jax.vmap(dir_ok)(actions)
        return (cap_mask32 & dok).astype(jnp.bool_)

    def takasa_actions(_):
        # Front row, non-singleton, not mtaji-moja protected col
        def eligible(action):
            row = jnp.where(action < 16, jnp.int32(0), jnp.int32(1))
            rem = jnp.where(action < 16, action, action - jnp.int32(16))
            col = rem // 2
            direction = rem % 2
            singleton = board[row, col] <= 1
            # Only front row for takasa (back row allowed only if front empty, which = loss)
            is_front = row == 0
            # Mtaji-moja: can't sow the hole that is the opponent's only target
            # i.e. can't sow front-row col c such that ending there would eliminate opp's last mtaji
            # The restriction is: if opp has only 1 occupied front hole at col c2,
            # and board[0, c2] == this action's source... actually mtaji-moja means
            # we can't sow our hole if doing so would expose the opponent's only mtaji.
            # Simpler: during takasa we can't sow board[0] hole that is the mtaji-moja target itself.
            # The rule: "the hole that is the only mtaji left for your opponent may not be sown."
            # That hole is board[0, c] such that board[2, c] > 0 and opp_count == 1.
            only_opp_col = jnp.argmax(board[2])  # col of opponent's sole occupied hole
            is_moja_source = mtaji_moja_active & is_front & (col == only_opp_col)
            is_special = is_kichwa_or_kimbi(col)
            fdir = forced_direction(col)
            dir_ok = (~is_special) | (fdir == direction)
            return is_front & (~singleton) & (~is_moja_source) & dir_ok

        actions = jnp.arange(32, dtype=jnp.int32)
        return jax.vmap(eligible)(actions).astype(jnp.bool_)

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
