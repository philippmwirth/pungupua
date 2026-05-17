"""Bao la Kiswahili — pgx-compatible Game class."""

from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from .board import flip_board, action_to_row_col_dir, make_observation, next_pos
from .rules import (
    namua_step,
    mtaji_step,
    compute_legal_action_mask,
    deactivate_nyumba_if_sowed,
)
from .sowing import sow
from .state import (
    GameState,
    INIT_BOARD,
    NYUMBA_STOP,
    NYUMBA_COL,
)


class Game:
    def init(self) -> GameState:
        return GameState(
            board=INIT_BOARD,
            stock=jnp.array([22, 22], jnp.int16),
            current_player=jnp.int32(0),
            stage=jnp.int32(0),
            winner=jnp.int32(-1),
            nyumba_active=jnp.array([True, True]),
            nyumba_pending=jnp.bool_(False),
            pending_direction=jnp.int32(1),
        )

    def step(self, state: GameState, action: Array) -> GameState:
        action = jnp.int32(action)

        # --- nyumba pending: player chooses stop or continue ---
        def nyumba_branch(s: GameState) -> GameState:
            def stop(_):
                b, na, stk = flip_board(s.board, s.nyumba_active, s.stock)
                new_state = s._replace(
                    board=b,
                    nyumba_active=na,
                    stock=stk,
                    current_player=1 - s.current_player,
                    nyumba_pending=jnp.bool_(False),
                )
                return _check_no_moves(new_state, s.current_player)

            def cont(_):
                # Pick up nyumba seeds, sow from next hole in the stored direction
                d = s.pending_direction
                b = s.board
                seeds = b[0, NYUMBA_COL].astype(jnp.int16)
                b = b.at[0, NYUMBA_COL].set(jnp.int16(0))
                nr, nc = next_pos(jnp.int32(0), jnp.int32(NYUMBA_COL), d)
                b, ny_pend, pend_dir, _ = sow(
                    b,
                    seeds,
                    nr,
                    nc,
                    d,
                    allow_capture=True,
                    nyumba_active=jnp.bool_(False),
                )
                na = s.nyumba_active.at[0].set(False)  # nyumba has been sowed out
                na = deactivate_nyumba_if_sowed(b, na, s.board)

                b_flip2, na_flip2, stk_flip2 = flip_board(b, na, s.stock)
                b = jax.lax.select(ny_pend, b, b_flip2)
                na = jax.lax.select(ny_pend, na, na_flip2)
                stk = jax.lax.select(ny_pend, s.stock, stk_flip2)
                new_cp = jnp.where(
                    ny_pend, s.current_player, jnp.int32(1 - s.current_player)
                )
                new_state = s._replace(
                    board=b,
                    nyumba_active=na,
                    stock=stk,
                    current_player=new_cp,
                    nyumba_pending=ny_pend,
                    pending_direction=pend_dir,
                )
                opp_front_empty = (b[2] == 0).all()
                winner = jax.lax.select(
                    opp_front_empty, s.current_player, jnp.int32(-1)
                )
                new_state = new_state._replace(winner=winner)
                return _check_no_moves(new_state, s.current_player)

            return jax.lax.cond(action == NYUMBA_STOP, stop, cont, None)

        # --- normal move ---
        def normal_branch(s: GameState) -> GameState:
            row, col, direction = action_to_row_col_dir(action)
            prev_board = s.board

            def namua(st: GameState):
                b, na, stk, ny_pend, pend_dir, ny_emp = namua_step(
                    st.board, st.stock, st.nyumba_active, col, direction
                )
                na = deactivate_nyumba_if_sowed(b, na, prev_board, ny_emp)
                new_stage = jnp.where(
                    (stk[0] == 0) & (st.stock[1] == 0), jnp.int32(1), st.stage
                )
                return b, na, stk, ny_pend, pend_dir, new_stage

            def mtaji(st: GameState):
                b, na, ny_pend, pend_dir, ny_emp = mtaji_step(
                    st.board, st.nyumba_active, row, col, direction
                )
                na = deactivate_nyumba_if_sowed(b, na, prev_board, ny_emp)
                return b, na, st.stock, ny_pend, pend_dir, st.stage

            b, na, stk, ny_pend, pend_dir, new_stage = jax.lax.cond(
                s.stage == 0, namua, mtaji, s
            )

            opp_front_empty = (b[2] == 0).all()
            winner = jax.lax.select(opp_front_empty, s.current_player, jnp.int32(-1))

            b_flip, na_flip, stk_flip = flip_board(b, na, stk)
            b_out = jax.lax.select(ny_pend, b, b_flip)
            na_out = jax.lax.select(ny_pend, na, na_flip)
            stk_out = jax.lax.select(ny_pend, stk, stk_flip)
            new_cp = jnp.where(
                ny_pend, s.current_player, jnp.int32(1 - s.current_player)
            )

            new_state = s._replace(
                board=b_out,
                stock=stk_out,
                current_player=new_cp,
                stage=new_stage,
                winner=winner,
                nyumba_active=na_out,
                nyumba_pending=ny_pend,
                pending_direction=pend_dir,
            )
            return _check_no_moves(new_state, s.current_player)

        return jax.lax.cond(
            state.nyumba_pending,
            nyumba_branch,
            normal_branch,
            state,
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.current_player
        return make_observation(
            state.board, state.nyumba_active, color, state.current_player
        )

    def legal_action_mask(self, state: GameState) -> Array:
        return compute_legal_action_mask(
            state.board,
            state.stock,
            state.stage,
            state.nyumba_active,
            state.nyumba_pending,
        )

    def is_terminal(self, state: GameState) -> Array:
        return state.winner >= 0

    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1.0, -1.0]).at[state.winner].set(1.0),
            jnp.zeros(2, jnp.float32),
        )


def _check_no_moves(state: GameState, prev_player: Array) -> GameState:
    """Set winner to prev_player if the new current player has no legal moves."""
    no_moves = ~compute_legal_action_mask(
        state.board,
        state.stock,
        state.stage,
        state.nyumba_active,
        state.nyumba_pending,
    ).any()
    winner = jax.lax.select(
        (state.winner < 0) & (~state.nyumba_pending) & no_moves,
        prev_player,
        state.winner,
    )
    return state._replace(winner=winner)
