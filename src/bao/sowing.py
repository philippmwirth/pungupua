"""Core sowing while_loop kernel."""

import jax
import jax.numpy as jnp
from jax import Array

from .board import next_pos, kichwa_col
from .state import NYUMBA_COL

MAX_ITER = 256


def sow(
    board: Array,
    seeds: Array,
    row: Array,
    col: Array,
    direction: Array,
    allow_capture: bool | Array = True,
    nyumba_active: bool | Array = True,
):
    """Sow `seeds` starting at (row, col) going in `direction`.

    Returns (new_board, nyumba_pending, pending_direction, nyumba_emptied).
    `nyumba_pending` is True if the last seed landed in an *active* nyumba with
    an empty opposing hole and the nyumba contains >=6 seeds.
    `nyumba_emptied` is True if the nyumba hole (row 0, NYUMBA_COL) was emptied
    via `do_continue` at any point during the sow (even if it was refilled
    afterwards by the same chain).

    Sowing rules (verified against diagrams 14-18):
      - Last seed in empty hole           -> done.
      - Last seed in occupied front hole,
        opposing occupied                 -> capture: take opposing seeds,
                                             re-enter from kichwa, repeat.
      - Last seed in occupied hole,
        no capture possible               -> pick up seeds, continue from
                                             next hole (same direction).
      - Last seed in nyumba, opp empty,
        nyumba active                     -> set nyumba_pending, stop loop.

    `allow_capture` is False during takasa moves.
    `nyumba_active` is False when the current player's nyumba is inactive
    (i.e. landing there is treated as a normal occupied hole).
    """
    nyumba_active = jnp.bool_(nyumba_active)

    # Carry: (board, seeds, row, col, direction, done, nyumba_pending,
    #        nyumba_emptied, _iter)
    def cond(carry):
        _, _, _, _, _, done, _, _, it = carry
        return (~done) & (it < MAX_ITER)

    def body(carry):
        b, s, r, c, d, done, ny_pend, ny_emp, it = carry

        # --- place one seed ---
        b = b.at[r, c].add(jnp.int16(1))
        s = s - jnp.int16(1)

        last = s == 0
        was_occupied = b[r, c] > jnp.int16(1)  # >1 because we just added 1

        opp_has_seeds = (r == 0) & (b[2, c] > 0)
        can_capture = allow_capture & opp_has_seeds & last & was_occupied

        # Nyumba stop/continue applies only when the nyumba has the special
        # status: active AND contains >= 6 seeds (RULES §The Nyumba).
        is_nyumba = (
            (r == 0)
            & (c == NYUMBA_COL)
            & last
            & was_occupied
            & (~opp_has_seeds)
            & nyumba_active
            & (b[r, c] >= jnp.int16(6))
        )

        # --- branch 1: capture ---
        def do_capture(args):
            b_, d_ = args
            captured = b_[2, c].astype(jnp.int16)
            b_ = b_.at[2, c].set(jnp.int16(0))
            # re-enter from appropriate kichwa
            kc = kichwa_col(d_)
            return b_, captured, jnp.int32(0), kc

        # --- branch 2: continue (pick up from current hole) ---
        def do_continue(args):
            b_, d_ = args
            picked = b_[r, c].astype(jnp.int16)
            b_ = b_.at[r, c].set(jnp.int16(0))
            nr, nc = next_pos(r, c, d_)
            return b_, picked, nr, nc

        # --- branch 3: done (empty hole or last seed with no action) ---
        def do_done(args):
            b_, d_ = args
            nr, nc = next_pos(r, c, d_)
            return b_, jnp.int16(0), nr, nc

        # Decide what happens when we placed the last seed
        # Priority: capture > nyumba_pending > continue > done
        def on_last(args):
            b_, d_ = args
            # capture?
            b2, s2, nr, nc = jax.lax.cond(
                can_capture,
                do_capture,
                lambda a: jax.lax.cond(
                    was_occupied & (~is_nyumba),
                    do_continue,
                    do_done,
                    a,
                ),
                (b_, d_),
            )
            new_done = (~can_capture) & ((~was_occupied) | is_nyumba)
            new_ny = is_nyumba
            return b2, s2, nr, nc, new_done, new_ny

        def on_not_last(args):
            b_, d_ = args
            nr, nc = next_pos(r, c, d_)
            return b_, s, nr, nc, jnp.bool_(False), jnp.bool_(False)

        b, s, nr, nc, new_done, new_ny = jax.lax.cond(
            last,
            on_last,
            on_not_last,
            (b, d),
        )

        # do_continue fires when: last seed, hole was occupied, not a capture,
        # not a (legal) nyumba_pending stop.  If that happens at the nyumba hole,
        # we have just picked up its seeds — mark it emptied.
        did_continue_at_nyumba = (
            last
            & was_occupied
            & (~is_nyumba)
            & (~can_capture)
            & (r == 0)
            & (c == NYUMBA_COL)
        )
        new_ny_emp = ny_emp | did_continue_at_nyumba

        return (
            b,
            s,
            nr.astype(jnp.int32),
            nc.astype(jnp.int32),
            d,
            new_done,
            new_ny,
            new_ny_emp,
            it + 1,
        )

    init_carry = (
        board,
        seeds.astype(jnp.int16),
        row.astype(jnp.int32),
        col.astype(jnp.int32),
        direction.astype(jnp.int32),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.int32(0),
    )
    final = jax.lax.while_loop(cond, body, init_carry)
    new_board, _, _, _, pending_dir, _, nyumba_pending, nyumba_emptied, _ = final
    return new_board, nyumba_pending, pending_dir.astype(jnp.int32), nyumba_emptied


def simulate_sow_ends_in_capture(
    board: Array,
    row: Array,
    col: Array,
    direction: Array,
    nyumba_active: bool | Array = True,
) -> Array:
    """Return True if sowing (row, col) in direction produces a capture.

    Used by legal_action_mask in mtaji to detect valid capture moves.
    Seeds must be >= 2 (singleton check done by caller).

    Implementation: runs the real `sow()` on a temp board with the source
    cleared and checks whether `board[2]` decreased anywhere along the way.
    This correctly accounts for `do_continue` chains that eventually lead to
    a capture, which the previous hand-rolled simulator missed.
    """
    seeds = board[row, col].astype(jnp.int16)
    new_board = board.at[row, col].set(jnp.int16(0))
    # If the source is the (current player's) nyumba, it's been sown out — for
    # the rest of this move it behaves as an ordinary hole, so disable the
    # nyumba_pending interrupt that would otherwise fire on wraparound.
    src_is_nyumba = (row == jnp.int32(0)) & (col == jnp.int32(NYUMBA_COL))
    eff_active = jnp.bool_(nyumba_active) & (~src_is_nyumba)
    nr, nc = next_pos(row, col, direction)
    after_board, _, _, _ = sow(
        new_board,
        seeds,
        nr,
        nc,
        direction,
        allow_capture=True,
        nyumba_active=eff_active,
    )
    return after_board[2].sum() < board[2].sum()
