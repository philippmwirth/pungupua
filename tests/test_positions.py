"""Tests using constructed board positions.

Covers areas not exercised by test_diagrams.py / test_game.py:
  - observe: one-hot encoding, nyumba channels, opponent perspective
  - is_terminal / rewards: explicit winner states
  - Stage transition to mtaji
  - Stock tracking and board-perspective flip after step
  - Namua mask: back-row always absent, capture mandatory, kichwa/kimbi for takasa
  - Mtaji mask: capture mandatory, back-row both dirs (Bug-2 regression), moja rule
  - Win by clearing the opponent's front row
  - Seed conservation for the nyumba 2-seed special move
"""

import jax.numpy as jnp
from bao import Game
from bao.state import GameState

g = Game()

Z8 = [0] * 8  # convenience: empty row


def st(board, *, stock=(0, 0), stage=1, na=(False, False), winner=-1):
    """Build a GameState with current_player=0 and no nyumba_pending."""
    return GameState(
        board=jnp.array(board, jnp.int16),
        stock=jnp.array(stock, jnp.int16),
        current_player=jnp.int32(0),
        stage=jnp.int32(stage),
        winner=jnp.int32(winner),
        nyumba_active=jnp.array(na),
        nyumba_pending=jnp.bool_(False),
        pending_direction=jnp.int32(1),
    )


# ---------------------------------------------------------------------------
# observe – one-hot encoding
# ---------------------------------------------------------------------------


def test_observe_one_hot_encodes_every_cell():
    """obs[:,:,:65] must be a valid one-hot of the actual seed counts."""
    board = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 9],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    s = st(board, na=(True, False))
    obs = g.observe(s)
    b = jnp.array(board, jnp.int16)
    for r in range(4):
        for c in range(8):
            k = int(b[r, c])
            assert float(obs[r, c, k]) == 1.0, f"({r},{c}): expected one-hot at bin {k}"
            assert float(obs[r, c, :65].sum()) == 1.0, (
                f"({r},{c}): one-hot must sum to 1"
            )


def test_observe_zero_seed_cell_uses_bin_zero():
    """An empty hole must have one-hot index 0 active."""
    s = st([[0] * 8] * 4)
    obs = g.observe(s)
    assert (obs[:, :, 0] == 1.0).all(), "all empty → every cell one-hot at bin 0"
    assert (obs[:, :, 1:65] == 0.0).all()


# ---------------------------------------------------------------------------
# observe – nyumba flag channels (65 = cur player, 66 = opponent)
# ---------------------------------------------------------------------------


def test_observe_nyumba_active_sets_channel_65():
    s = st([[0] * 8] * 4, na=(True, False))
    obs = g.observe(s)
    assert (obs[:, :, 65] == 1.0).all(), "ch65 must be 1 when cur nyumba active"
    assert (obs[:, :, 66] == 0.0).all(), "ch66 must be 0 when opp nyumba inactive"


def test_observe_nyumba_inactive_clears_channel_65():
    s = st([[0] * 8] * 4, na=(False, True))
    obs = g.observe(s)
    assert (obs[:, :, 65] == 0.0).all(), "ch65 must be 0 when cur nyumba inactive"
    assert (obs[:, :, 66] == 1.0).all(), "ch66 must be 1 when opp nyumba active"


# ---------------------------------------------------------------------------
# observe – opponent perspective (color != current_player)
# ---------------------------------------------------------------------------


def test_observe_opponent_sees_flipped_board():
    """observe(color=1) shows the board from the opponent's point of view:
    their row 0 = reversed board[2], their row 2 = reversed board[0]."""
    board = [
        [1, 0, 0, 0, 0, 0, 0, 0],  # cur front: 1 seed at col 0
        Z8,
        [0, 0, 0, 0, 0, 0, 0, 9],  # opp front: 9 seeds at col 7
        Z8,
    ]
    s = st(board)
    obs1 = g.observe(s, jnp.int32(1))
    # Opponent row 0 = board[2,::-1]: col 0 should show 9 seeds
    assert float(obs1[0, 0, 9]) == 1.0, "opp row-0 col-0 should reflect board[2,7]=9"
    # Opponent row 2 = board[0,::-1]: col 7 should show 1 seed
    assert float(obs1[2, 7, 1]) == 1.0, "opp row-2 col-7 should reflect board[0,0]=1"


def test_observe_current_player_view_unchanged():
    """observe(color=current_player) must equal observe() with no color argument."""
    board = [
        [0, 3, 0, 0, 0, 0, 0, 0],
        Z8,
        [0, 0, 0, 5, 0, 0, 0, 0],
        Z8,
    ]
    s = st(board)
    obs_default = g.observe(s)
    obs_color0 = g.observe(s, jnp.int32(0))
    assert (obs_default == obs_color0).all(), (
        "color=0 should give the same view as default"
    )


def test_observe_nyumba_flags_swapped_for_opponent():
    """When observing as the opponent, channels 65 and 66 must swap."""
    s = st([[0] * 8] * 4, na=(True, False))
    obs0 = g.observe(s, jnp.int32(0))
    obs1 = g.observe(s, jnp.int32(1))
    # color=0: ch65 = nyumba_active[0]=True, ch66 = nyumba_active[1]=False
    assert (obs0[:, :, 65] == 1.0).all()
    assert (obs0[:, :, 66] == 0.0).all()
    # color=1: perspective swapped: ch65 = nyumba_active[1]=False, ch66 = nyumba_active[0]=True
    assert (obs1[:, :, 65] == 0.0).all()
    assert (obs1[:, :, 66] == 1.0).all()


# ---------------------------------------------------------------------------
# is_terminal / rewards
# ---------------------------------------------------------------------------


def test_is_terminal_false_for_ongoing_game():
    s = st([[2] * 8, Z8, [2] * 8, Z8])
    assert not g.is_terminal(s)


def test_is_terminal_true_when_winner_set():
    for w in (0, 1):
        assert g.is_terminal(st([[0] * 8] * 4, winner=w)), (
            f"winner={w} must be terminal"
        )


def test_rewards_zero_when_not_terminal():
    s = st([[2] * 8, Z8, [2] * 8, Z8])
    r = g.rewards(s)
    assert float(r[0]) == 0.0 and float(r[1]) == 0.0


def test_rewards_winner_0():
    r = g.rewards(st([[0] * 8] * 4, winner=0))
    assert float(r[0]) == 1.0 and float(r[1]) == -1.0


def test_rewards_winner_1():
    r = g.rewards(st([[0] * 8] * 4, winner=1))
    assert float(r[0]) == -1.0 and float(r[1]) == 1.0


def test_win_by_clearing_opponent_front_row():
    """An mtaji capture that empties board[2] must set winner = current_player (0)
    within the same step.

    Setup: board[0,0]=3 (left kichwa, forced right), board[0,3]=1 (occupied),
    board[2,3]=2 (opponent's only front hole).
    Action 1 = col-0 right: sow 3 seeds, last lands at (0,3), captures board[2,3]=2,
    leaving board[2] all zeros.
    """
    s = st(
        [
            [3, 0, 0, 1, 0, 0, 0, 0],
            Z8,
            [0, 0, 0, 2, 0, 0, 0, 0],
            Z8,
        ]
    )
    s2 = g.step(s, jnp.int32(1))
    assert g.is_terminal(s2), "should be terminal after clearing opp front row"
    assert int(s2.winner) == 0, f"winner should be 0, got {s2.winner}"


# ---------------------------------------------------------------------------
# Stage transition and stock tracking
# ---------------------------------------------------------------------------


def test_stage_transitions_to_mtaji_when_both_stocks_reach_zero():
    """stage flips 0→1 on the namua move that plays the very last stock seed."""
    # stock=(1,0): current player uses last seed; opponent already at 0 → both zero.
    s = st([[0, 0, 0, 0, 0, 2, 0, 0], Z8, Z8, Z8], stock=(1, 0), stage=0)
    s2 = g.step(s, jnp.int32(11))  # col5 right takasa
    assert int(s2.stage) == 1, (
        f"stage should be 1 after last stock seed, got {s2.stage}"
    )


def test_stage_stays_0_while_stock_remains():
    """stage must stay 0 when the current player still has stock left."""
    s = st([[0, 0, 0, 0, 0, 2, 0, 0], Z8, Z8, Z8], stock=(3, 0), stage=0)
    s2 = g.step(s, jnp.int32(11))
    assert int(s2.stage) == 0, f"stage should remain 0 while stock > 1, got {s2.stage}"


def test_stock_decrements_by_one_per_namua_move():
    """Each namua move consumes exactly one stock seed."""
    s = st([[0, 0, 0, 0, 0, 2, 0, 0], Z8, Z8, Z8], stock=(3, 5), stage=0)
    s2 = g.step(s, jnp.int32(11))  # col5 right
    # After flip: new player sees stock = [old_opp=5, old_cur-1=2]
    assert int(s2.stock[0]) == 5, "new player's stock should be 5 (was opp 5)"
    assert int(s2.stock[1]) == 2, "old player's remaining stock should be 2"


# ---------------------------------------------------------------------------
# Board perspective and current_player tracking
# ---------------------------------------------------------------------------


def test_current_player_alternates_each_step():
    s = g.init()
    assert int(s.current_player) == 0
    s2 = g.step(s, jnp.where(g.legal_action_mask(s))[0][0])
    assert int(s2.current_player) == 1
    s3 = g.step(s2, jnp.where(g.legal_action_mask(s2))[0][0])
    assert int(s3.current_player) == 0


def test_board_flips_to_new_players_perspective_after_step():
    """After a non-capturing step the new current player's row 0 equals the
    reversed old opponent front row (board[2]).  The init position has no capture
    available on the first move, so board[2] is guaranteed unchanged."""
    s = g.init()
    old_opp_front = s.board[2]  # [0,2,2,6,0,0,0,0]
    s2 = g.step(s, jnp.int32(9))  # col4 right – no capture at init
    assert jnp.array_equal(s2.board[0], old_opp_front[::-1]), (
        f"s2.board[0] should be reversed old board[2], got {s2.board[0]}"
    )


# ---------------------------------------------------------------------------
# Namua legal mask
# ---------------------------------------------------------------------------


def test_namua_back_row_actions_never_legal():
    """Actions 16-31 (back row) must be absent from every namua legal mask."""
    s = g.init()
    mask = g.legal_action_mask(s)
    assert not mask[16:32].any(), "back-row actions must not appear in namua"


def test_namua_capture_mandatory_blocks_takasa():
    """When a capture is available, non-capturing takasa holes must be absent."""
    # col5 has seeds on both sides → capture mandatory.
    # col2 only has seeds on our side → would be a takasa, but must be blocked.
    s = st(
        [[0, 0, 2, 0, 0, 3, 0, 0], Z8, [0, 0, 0, 0, 0, 5, 0, 0], Z8],
        stock=(1, 0),
        stage=0,
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert {10, 11}.issubset(legal), "col5 capture must be legal in both directions"
    assert 4 not in legal, "col2 left takasa must be blocked (capture mandatory)"
    assert 5 not in legal, "col2 right takasa must be blocked"


def test_namua_takasa_from_kichwa_kimbi_is_unconstrained():
    """The kichwa/kimbi forced-direction rule applies ONLY when capturing
    from those holes (RULES §Kichwa and Kimbi).  A plain takasa from col 0,
    1, 6, or 7 may go either direction.
    """
    # All four kichwa/kimbi cols occupied, no opp seeds anywhere -> takasa.
    s = st([[3, 3, 0, 0, 0, 0, 3, 3], Z8, Z8, Z8], stock=(1, 0), stage=0)
    mask = g.legal_action_mask(s)
    for c in (0, 1, 6, 7):
        assert bool(mask[c * 2 + 0]), f"col{c}-left takasa must be legal (no capture)"
        assert bool(mask[c * 2 + 1]), f"col{c}-right takasa must be legal (no capture)"


def test_namua_capture_from_kichwa_kimbi_is_constrained():
    """When capturing from a kichwa/kimbi hole, the direction is still forced."""
    # Capture available at col0 (left kichwa): only right is legal.
    s = st(
        [[3, 0, 0, 0, 0, 0, 0, 0], Z8, [2, 0, 0, 0, 0, 0, 0, 0], Z8],
        stock=(1, 0),
        stage=0,
    )
    mask = g.legal_action_mask(s)
    assert not mask[0], "col0-left must be illegal for left kichwa capture"
    assert mask[1], "col0-right must be legal for left kichwa capture"


def test_namua_right_kichwa_forced_left_for_capture():
    """Right kichwa (col7) capture must only allow the leftward direction."""
    s = st(
        [[0, 0, 0, 0, 0, 0, 0, 3], Z8, [0, 0, 0, 0, 0, 0, 0, 5], Z8],
        stock=(1, 0),
        stage=0,
    )
    mask = g.legal_action_mask(s)
    assert mask[14], "col7-left must be legal for right kichwa capture"
    assert not mask[15], "col7-right must be illegal for right kichwa"


# ---------------------------------------------------------------------------
# Namua mask – Difference #5: singleton takasa source
# ---------------------------------------------------------------------------


def test_namua_takasa_singleton_source_is_legal():
    """Difference #5: in namua, a singleton front-row hole IS a legal takasa
    source.  The stock seed added before sowing lifts it to 2, so the
    "never sow a singleton" rule (which applies in mtaji) does not apply here.
    """
    # col3 holds a single seed, no opponent seeds anywhere → takasa, singleton source.
    s = st(
        [[0, 0, 0, 1, 0, 0, 0, 0], Z8, Z8, Z8],
        stock=(1, 0), stage=0, na=(False, False),
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert 6 in legal and 7 in legal, (
        "a singleton must be a legal namua takasa source in both directions"
    )


def test_namua_takasa_singleton_source_execution():
    """Executing a namua takasa from a singleton: the stock seed makes it 2, both
    seeds are picked up and sown one per hole.

    col3=1, stock seed → 2, sow right from col3: land at col4 (1) and col5 (1),
    last seed in empty col5 → move ends.  Seeds are conserved.
    """
    s = st(
        [[0, 0, 0, 1, 0, 0, 0, 0], Z8, Z8, Z8],
        stock=(1, 0), stage=0, na=(False, False),
    )
    total_before = int(s.board.sum()) + int(s.stock.sum())
    s2 = g.step(s, jnp.int32(7))  # col3 right
    your_front = s2.board[2, ::-1]  # recover your front from opponent's view
    assert jnp.array_equal(your_front, jnp.array([0, 0, 0, 0, 1, 1, 0, 0], jnp.int16)), (
        f"singleton takasa should sow 2 seeds into col4/col5, got {your_front.tolist()}"
    )
    total_after = int(s2.board.sum()) + int(s2.stock.sum())
    assert total_before == total_after, (
        f"seed count changed: {total_before} → {total_after}"
    )


# ---------------------------------------------------------------------------
# Namua mask – Difference #1: takasa entry into the nyumba
# ---------------------------------------------------------------------------


def test_namua_takasa_into_functional_nyumba_blocked():
    """Difference #1: on a non-capturing move the seed may not be entered into a
    functional nyumba (>= 6 seeds, active) while another front hole is occupied.
    """
    # nyumba (col4) = 6 and active, plus col5 = 3.  No opp seeds -> takasa.
    s = st(
        [[0, 0, 0, 0, 6, 3, 0, 0], Z8, Z8, Z8],
        stock=(1, 0), stage=0, na=(True, False),
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert 8 not in legal and 9 not in legal, (
        "takasa entry into a functional nyumba must be blocked (cols 8/9)"
    )
    assert 10 in legal and 11 in legal, "other front holes stay legal for takasa"


def test_namua_takasa_into_nyumba_allowed_when_only_hole():
    """Difference #1 exception: a functional nyumba may be entered on a takasa
    move when it is the only occupied front hole (the >= 6 special 2-seed rule).
    """
    s = st(
        [[0, 0, 0, 0, 7, 0, 0, 0], Z8, Z8, Z8],
        stock=(1, 0), stage=0, na=(True, False),
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert 8 in legal or 9 in legal, (
        "nyumba must be playable when it is the only occupied front hole"
    )


def test_namua_takasa_into_nonfunctional_nyumba_allowed():
    """A nyumba with < 6 seeds is an ordinary hole: takasa entry is allowed even
    when other holes are occupied (Difference #1 only restricts functional ones).
    """
    s = st(
        [[0, 0, 0, 0, 5, 3, 0, 0], Z8, Z8, Z8],
        stock=(1, 0), stage=0, na=(True, False),
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert 8 in legal and 9 in legal, (
        "takasa into a < 6 nyumba (ordinary hole) must be allowed"
    )


def test_namua_capture_into_functional_nyumba_still_allowed():
    """Difference #1 restricts only non-capturing entry.  Entering the nyumba to
    make a capture remains legal even with other holes occupied.
    """
    # nyumba (col4) = 6 active with an opposing hole -> capture available there.
    s = st(
        [[0, 0, 0, 0, 6, 3, 0, 0], Z8, [0, 0, 0, 0, 2, 0, 0, 0], Z8],
        stock=(1, 0), stage=0, na=(True, False),
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert 8 in legal or 9 in legal, (
        "capturing entry into the nyumba must remain legal"
    )


# ---------------------------------------------------------------------------
# Mtaji legal mask
# ---------------------------------------------------------------------------


def test_mtaji_capture_mandatory_blocks_takasa():
    """In mtaji, a capturing move existing must suppress all takasa actions.

    board[0,0]=3 (left kichwa, forced right), board[0,3]=1 (occupied),
    board[2,3]=3 (opponent seeds) → sowing col0 right lands last seed at (0,3),
    producing a capture.  board[0,2]=2 would be a valid takasa, but capture is
    mandatory so it must be absent.
    """
    s = st(
        [[3, 0, 2, 1, 0, 0, 0, 0], Z8, [0, 0, 0, 3, 0, 0, 0, 0], Z8],
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert 1 in legal, "col0-right capture must be legal"
    assert 4 not in legal, "col2-left takasa must be blocked"
    assert 5 not in legal, "col2-right takasa must be blocked"


def test_mtaji_back_row_col1_left_legal_when_capture():
    """Back-row col1 left (action 18) must be legal when it produces a capture.

    This is a regression test for Bug 2: back-row holes at cols 0/1/6/7 must NOT
    inherit the front-row kichwa/kimbi direction constraint.

    Setup: board[1,1]=7.  Sowing left from (1,1), the 7 seeds travel through
    back-row cols 2→3→4→5→6→7, then wrap to front-row col 7.  The last seed
    lands at (0,7); board[0,7]=1 (occupied) and board[2,7]=3 → capture.
    """
    s = st(
        [
            [0, 0, 0, 0, 0, 0, 0, 1],  # board[0,7]=1 (occupied)
            [0, 7, 0, 0, 0, 0, 0, 0],  # board[1,1]=7
            [0, 0, 0, 0, 0, 0, 0, 3],  # board[2,7]=3
            Z8,
        ],
    )
    mask = g.legal_action_mask(s)
    assert mask[18], (
        "action 18 (B2L) must be legal – back-row col1 left produces a capture"
    )


def test_mtaji_back_row_col6_right_legal_when_capture():
    """Back-row col6 right (action 29) must be legal when it produces a capture.

    board[1,6]=7, board[0,0]=1 (occupied), board[2,0]=3.
    Sowing (1,6) right: next pos goes back-row 5→4→3→2→1→0, then front-row 0.
    7th seed lands at (0,0); board[0,0]=1+1=2 (occupied), board[2,0]=3 → capture.
    """
    # Sowing (1,6) right: next_pos(1,6,1): pos_right(1,6)=15-6=9. (9+1)%16=10. nc=15-10=5, nr=1 → (1,5).
    # 7 seeds: (1,5),(1,4),(1,3),(1,2),(1,1),(1,0),(0,0). Last at (0,0): board[0,0]=1+1=2, board[2,0]=3 → capture.
    s = st(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # board[0,0]=1
            [0, 0, 0, 0, 0, 0, 7, 0],  # board[1,6]=7
            [3, 0, 0, 0, 0, 0, 0, 0],  # board[2,0]=3
            Z8,
        ],
    )
    mask = g.legal_action_mask(s)
    # action 29 = 16 + 6*2 + 1 = 29
    assert mask[29], (
        "action 29 (B7R) must be legal – back-row col6 right produces a capture"
    )


def test_mtaji_moja_blocks_sole_opponent_col_in_takasa():
    """Mtaji-moja: when the opponent has exactly one occupied front hole (col C),
    our front hole at col C cannot be the source of a takasa move.

    board[0,2]=3 and board[0,3]=3.  board[2,3]=2 is the only opponent front hole.
    No capture is reachable from either hole, so we fall into takasa_actions.
    Moja blocks col3; col2 must still be legal.
    """
    s = st(
        [[0, 0, 3, 3, 0, 0, 0, 0], Z8, [0, 0, 0, 2, 0, 0, 0, 0], Z8],
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert 4 in legal, "col2-left takasa must be legal"
    assert 5 in legal, "col2-right takasa must be legal"
    assert 6 not in legal, "col3-left must be blocked by moja rule"
    assert 7 not in legal, "col3-right must be blocked by moja rule"


def test_mtaji_takasa_from_kichwa_kimbi_is_unconstrained():
    """Mtaji takasa from a kichwa/kimbi front-row hole has no direction
    constraint (same as namua takasa).
    """
    # Non-singleton seeds at all four kichwa/kimbi cols, no captures reachable.
    s = st([[3, 3, 0, 0, 0, 0, 3, 3], Z8, Z8, Z8], stock=(0, 0), stage=1)
    mask = g.legal_action_mask(s)
    for c in (0, 1, 6, 7):
        assert bool(mask[c * 2 + 0]), f"col{c}-left mtaji takasa must be legal"
        assert bool(mask[c * 2 + 1]), f"col{c}-right mtaji takasa must be legal"


# ---------------------------------------------------------------------------
# Mtaji – first-lap capture rule (#8)
# ---------------------------------------------------------------------------


def test_mtaji_first_lap_relay_then_loaded_is_takasa_not_capture():
    """A move whose first lap *relays* (lands in an occupied hole with an empty
    opposite) is a takasa even if a later relay lap reaches a loaded hole — Bao
    rule: "if the first lap doesn't capture, nothing will be captured."

    Sow col2 (2 seeds) right:
      lap 1 → col3, col4 (col4 occupied, board[2,4]=0 empty) → relay, NOT a capture
      lap 2 → would reach board[2,6]=3, but the move is already non-capturing.
    The move must be offered as a takasa (both directions) and capture nothing.
    """
    s = st([[0, 0, 2, 0, 1, 0, 1, 0], Z8, [0, 0, 0, 0, 0, 0, 3, 0], Z8])
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    # col2 is the only non-singleton front hole; offered as takasa both ways.
    assert {4, 5}.issubset(legal), "col2 must be a legal takasa source both directions"
    s2 = g.step(s, jnp.int32(5))  # col2 right
    opp_front = s2.board[0][::-1]  # opponent front row after the flip
    assert int(opp_front[6]) == 3, "the loaded hole must NOT be captured by a takasa"


def test_mtaji_genuine_first_lap_capture_still_mandatory():
    """A move whose first lap ends opposite a loaded hole is still a capture and
    remains mandatory (regression guard for the first-lap fix)."""
    # col0=3 right: lap 1 lands at (0,3) with board[2,3]=2 loaded → first-lap capture.
    s = st([[3, 0, 2, 1, 0, 0, 0, 0], Z8, [0, 0, 0, 2, 0, 0, 0, 0], Z8])
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert 1 in legal, "col0-right first-lap capture must be legal"
    assert 4 not in legal and 5 not in legal, (
        "col2 takasa must be blocked while a real capture exists"
    )


def test_mtaji_capture_via_do_continue_chain_is_not_a_capture():
    """A mtaji move whose *first lap* relays (do_continue) is NOT a capture, even
    if the continued chain would eventually reach a loaded hole.  Per the Bao
    first-lap rule the whole position is a takasa.

    Setup:
      board[0] = [2, 2, 1, 1, 1, 1, 1, 1]
      board[2] = [0, 0, 0, 0, 0, 0, 0, 2]    (opp seeds only at col7)
    Sow col1-right (2 seeds): place at col2 (occ), col3 (last, occ, opp empty)
    -> first lap relays, so the move captures nothing.  No hole has a first-lap
    capture, so this is a takasa: the non-singleton holes col0 and col1 are
    legal sources (both directions).
    """
    s = st(
        [[2, 2, 1, 1, 1, 1, 1, 1], Z8, [0, 0, 0, 0, 0, 0, 0, 2], Z8],
        stock=(0, 0),
        stage=1,
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    # Takasa situation: the two non-singleton holes are legal both directions.
    assert legal == {0, 1, 2, 3}, f"expected col0/col1 takasa moves, got {legal}"
    # Executing the chained move must capture nothing (opp col7 keeps its 2).
    s2 = g.step(s, jnp.int32(3))  # col1 right
    opp_front = s2.board[0][::-1]
    assert int(opp_front[7]) == 2, "first-lap relay must not capture later in the chain"


def test_mtaji_singleton_not_legal_as_takasa_source():
    """A front-row hole with exactly 1 seed must not appear in the mtaji mask."""
    # col2=2 added so both col2 and col3 are paired non-singletons → moja inactive.
    s = st(
        [[0, 1, 2, 3, 0, 0, 0, 0], Z8, [0, 0, 2, 2, 0, 0, 0, 0], Z8],
    )
    mask = g.legal_action_mask(s)
    assert not mask[2], "col1-left must be illegal (singleton)"
    assert not mask[3], "col1-right must be illegal (singleton)"
    assert mask[6] or mask[7], "col3 should be legal (non-singleton, no moja)"


def test_mtaji_takasa_back_row_fallback_when_no_front_row_moves():
    """When no captures exist and the front row has only singletons/empty holes,
    back-row non-singleton holes must be offered as takasa fallback.

    Front row: singletons only (col1=1, col3=1).
    Back row: col2=3, col4=5.
    Opponent front: seeds at cols not opposite any front-row non-singleton → no captures.
    Expected: B3L/B3R (actions 20/21) and B5L/B5R (actions 24/25) are legal;
    no front-row actions are legal.
    """
    s = st(
        [
            [0, 1, 0, 1, 0, 0, 0, 0],  # cur front: only singletons
            [0, 0, 3, 0, 5, 0, 0, 0],  # cur back: non-singletons at col2, col4
            [0, 0, 0, 0, 0, 2, 3, 0],  # opp front: no seeds opposite cur non-empty front cols
            Z8,
        ],
    )
    mask = g.legal_action_mask(s)
    legal = set(int(a) for a in jnp.where(mask)[0].tolist())

    # Back-row non-singletons must be offered
    assert 20 in legal, "B3L (action 20) must be legal as back-row takasa fallback"
    assert 21 in legal, "B3R (action 21) must be legal as back-row takasa fallback"
    assert 24 in legal, "B5L (action 24) must be legal as back-row takasa fallback"
    assert 25 in legal, "B5R (action 25) must be legal as back-row takasa fallback"

    # No front-row actions should be legal (all singletons or empty)
    front_legal = [a for a in legal if a < 16]
    assert not front_legal, f"no front-row actions should be legal, got {front_legal}"


def test_mtaji_takasa_front_row_preferred_over_back_row():
    """When no captures exist but both front-row and back-row non-singletons are
    available, only front-row moves should be offered (front row takes priority).

    Front row: col2=3.  Back row: col4=5.  No captures reachable.
    Expected: A3L/A3R legal; B5L/B5R not legal.
    """
    s = st(
        [
            [0, 0, 3, 0, 0, 0, 0, 0],  # cur front: col2=3
            [0, 0, 0, 0, 5, 0, 0, 0],  # cur back: col4=5
            Z8,                          # opp front: empty → no captures
            Z8,
        ],
    )
    mask = g.legal_action_mask(s)
    legal = set(int(a) for a in jnp.where(mask)[0].tolist())

    assert 4 in legal or 5 in legal, "A3L or A3R must be legal (front-row takasa)"
    assert 24 not in legal, "B5L must not be legal when front-row takasa is available"
    assert 25 not in legal, "B5R must not be legal when front-row takasa is available"


# ---------------------------------------------------------------------------
# Seed conservation for the nyumba 2-seed special move
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mid-chain direction flip at kichwa / kimbi
# ---------------------------------------------------------------------------


def test_mid_chain_capture_at_right_kichwa_flips_direction_to_left():
    """A mid-turn chain capture landing at col 7 (right kichwa) must force
    re-entry from col 7 going LEFT, not from col 0 going right.

    Setup (mtaji):
      board[0] = [0, 0, 0, 0, 0, 2, 0, 1]   (col5=2, col7=1)
      board[2] = [2, 0, 0, 0, 0, 0, 0, 3]   (col0=2 keeps opp front non-empty; col7=3)

    Action A6R (col5 right = action 11):
      - sow col5 (2 seeds) rightward → lands at col6 (1) then col7 (last, occupied,
        opp col7=3) → CAPTURE at right kichwa → direction flips to LEFT.
      - 3 captured seeds re-enter from col7 going left → land at col7, col6, col5.

    Expected final player front (pre-flip): [0,0,0,0,0,1,2,3]
    After board flip, new board[2] = old player front reversed = [3,2,1,0,0,0,0,0].

    With the old (buggy) code the captured seeds would have entered from col0 going
    right, giving [1,1,1,0,0,0,1,2] (pre-flip) and new board[2]=[2,1,0,0,0,1,1,1].
    """
    s = st(
        [
            [0, 0, 0, 0, 0, 2, 0, 1],  # cur front: col5=2, col7=1
            Z8,
            [2, 0, 0, 0, 0, 0, 0, 3],  # opp front: col0=2, col7=3
            Z8,
        ],
    )
    s2 = g.step(s, jnp.int32(11))  # col5 right
    assert int(s2.winner) < 0, "game should still be ongoing"
    expected = jnp.array([3, 2, 1, 0, 0, 0, 0, 0], jnp.int16)
    assert jnp.array_equal(s2.board[2], expected), (
        f"after direction flip, new board[2] should be {expected.tolist()}, "
        f"got {s2.board[2].tolist()}"
    )


def test_mid_chain_capture_at_left_kimbi_flips_direction_to_right():
    """A mid-turn chain capture landing at col 1 (left kimbi) must force
    re-entry from col 0 going RIGHT, not from col 7 going left.

    Setup (mtaji):
      board[0] = [0, 1, 0, 2, 0, 0, 0, 0]   (col1=1, col3=2)
      board[2] = [2, 3, 0, 0, 0, 0, 0, 0]   (col0=2; col1=3)

    Action A4L (col3 left = action 6):
      - sow col3 (2 seeds) leftward → col2 (1), col1 (last, occupied, opp col1=3)
        → CAPTURE at left kimbi → direction flips to RIGHT.
      - 3 captured seeds enter from col0 going right → col0, col1, col2
        (col2 occupied → do_continue → col3, col4 → empty → done).

    Expected final player front (pre-flip): [1,3,0,1,1,0,0,0]
    After flip, new board[2] = old player front reversed = [0,0,0,1,1,0,3,1].

    With the old (buggy) code the captured seeds would have entered from col7
    going left, giving [0,2,1,0,0,1,1,1] (pre-flip) and new board[2]=[1,1,1,0,0,1,2,0].
    """
    s = st(
        [
            [0, 1, 0, 2, 0, 0, 0, 0],  # cur front: col1=1, col3=2
            Z8,
            [2, 3, 0, 0, 0, 0, 0, 0],  # opp front: col0=2, col1=3
            Z8,
        ],
    )
    s2 = g.step(s, jnp.int32(6))  # col3 left
    assert int(s2.winner) < 0, "game should still be ongoing"
    expected = jnp.array([0, 0, 0, 1, 1, 0, 3, 1], jnp.int16)
    assert jnp.array_equal(s2.board[2], expected), (
        f"after direction flip, new board[2] should be {expected.tolist()}, "
        f"got {s2.board[2].tolist()}"
    )


def test_seed_conservation_nyumba_two_seed_move():
    """The nyumba-only 2-seed special move must not create or destroy seeds."""
    s = st(
        [[0, 0, 0, 0, 7, 0, 0, 0], Z8, Z8, Z8], stock=(1, 0), stage=0, na=(True, True)
    )
    total_before = int(s.board.sum()) + int(s.stock.sum())
    s2 = g.step(s, jnp.int32(8))  # nyumba left (the only valid direction here)
    total_after = int(s2.board.sum()) + int(s2.stock.sum())
    assert total_before == total_after, (
        f"seed count changed: {total_before} → {total_after}"
    )
