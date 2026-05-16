"""Core sowing while_loop kernel."""
import jax
import jax.numpy as jnp
from jax import Array

from .board import next_pos, kichwa_col
from .state import NYUMBA_COL

MAX_ITER = 256


def sow(board: Array, seeds: Array, row: Array, col: Array,
        direction: Array, allow_capture: bool | Array = True):
    """Sow `seeds` starting at (row, col) going in `direction`.

    Returns (new_board, nyumba_pending, pending_direction) where nyumba_pending
    is True if the last seed landed in the nyumba with an empty opposing hole.

    Sowing rules (verified against diagrams 14-18):
      - Last seed in empty hole           -> done.
      - Last seed in occupied front hole,
        opposing occupied                 -> capture: take opposing seeds,
                                             re-enter from kichwa, repeat.
      - Last seed in occupied hole,
        no capture possible               -> pick up seeds, continue from
                                             next hole (same direction).
      - Last seed in nyumba, opp empty    -> set nyumba_pending, stop loop.

    `allow_capture` is False during takasa moves.
    """
    # Carry: (board, seeds, row, col, direction, done, nyumba_pending, _iter)
    def cond(carry):
        _, _, _, _, _, done, _, it = carry
        return (~done) & (it < MAX_ITER)

    def body(carry):
        b, s, r, c, d, done, ny_pend, it = carry

        # --- place one seed ---
        b = b.at[r, c].add(jnp.int16(1))
        s = s - jnp.int16(1)

        last = s == 0
        was_occupied = b[r, c] > jnp.int16(1)   # >1 because we just added 1

        opp_has_seeds = (r == 0) & (b[2, c] > 0)
        can_capture = allow_capture & opp_has_seeds & last & was_occupied

        is_nyumba = (r == 0) & (c == NYUMBA_COL) & last & was_occupied & (~opp_has_seeds)

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
            new_done = (~can_capture) & (
                (~was_occupied) | is_nyumba
            )
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

        return b, s, nr.astype(jnp.int32), nc.astype(jnp.int32), d, new_done, new_ny, it + 1

    init_carry = (
        board,
        seeds.astype(jnp.int16),
        row.astype(jnp.int32),
        col.astype(jnp.int32),
        direction.astype(jnp.int32),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.int32(0),
    )
    final = jax.lax.while_loop(cond, body, init_carry)
    new_board, _, _, _, pending_dir, _, nyumba_pending, _ = final
    return new_board, nyumba_pending, pending_dir.astype(jnp.int32)


def simulate_sow_ends_in_capture(board: Array, row: Array, col: Array,
                                  direction: Array) -> Array:
    """Return True if sowing (row, col) in direction produces a capture.

    Used by legal_action_mask in mtaji to detect valid capture moves.
    Seeds must be >= 2 (singleton check done by caller).
    """
    seeds = board[row, col].astype(jnp.int16)

    # Carry: (row, col, seeds, captured)
    def cond(carry):
        _, _, s, captured, it = carry
        return (s > 0) & (~captured) & (it < MAX_ITER)

    def body(carry):
        r, c, s, captured, it = carry
        s = s - jnp.int16(1)
        last = s == 0
        nr, nc = next_pos(r, c, direction)

        seed_after = board[r, c] + jnp.int16(1)   # hypothetical
        opp_seeds = board[2, c]
        capture_here = last & (r == 0) & (seed_after > 1) & (opp_seeds > 0)

        r = jnp.where(last & ~capture_here & (seed_after > 1), r, nr)
        c = jnp.where(last & ~capture_here & (seed_after > 1), c, nc)

        return r, c, s, capture_here, it + 1

    init = (row.astype(jnp.int32), col.astype(jnp.int32),
            seeds, jnp.bool_(False), jnp.int32(0))
    _, _, _, captured, _ = jax.lax.while_loop(cond, body, init)
    return captured
