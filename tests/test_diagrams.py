"""Unit tests reproducing diagrams from RULES.md."""
import jax.numpy as jnp
import pytest
from bao import Game
from bao.state import GameState


def make_state(board, stock=(10, 10), nyumba_active=(True, True), stage=0):
    return GameState(
        board=jnp.array(board, jnp.int16),
        stock=jnp.array(stock, jnp.int16),
        current_player=jnp.int32(0),
        stage=jnp.int32(stage),
        winner=jnp.int32(-1),
        nyumba_active=jnp.array(nyumba_active),
        nyumba_pending=jnp.bool_(False),
        pending_direction=jnp.int32(1),
    )


g = Game()


def test_init():
    s = g.init()
    expected = jnp.array([
        [0, 0, 0, 0, 6, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], jnp.int16)
    assert jnp.array_equal(s.board, expected)
    assert jnp.array_equal(s.stock, jnp.array([22, 22], jnp.int16))


def test_diagram3_legal_actions():
    """Diagram 3: only col4 is capturable."""
    s = make_state([
        [0, 0, 0, 0, 7, 0, 2, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 8, 1, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    # col4, both directions (col4 is not kichwa/kimbi)
    assert legal == {8, 9}


def test_diagram3_to_4_capture_left():
    """Diagram 3->4: capture col4, enter 1 seed at left kichwa (empty) -> done."""
    s = make_state([
        [0, 0, 0, 0, 7, 0, 2, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 8, 1, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    # action 8 = col4 left; kichwa for left-sow is col7 (right kichwa)
    # captured 1 seed sowed from right kichwa going left:
    #   col7 was 1+1=2? No — col7 was 1. Sow 1 seed: place at col7 (1->2). last, was_occupied.
    #   opp col7=0 -> not capture. nyumba? col7 != NYUMBA_COL=4. continue: pick up 2, sow from col6.
    # Hmm, actually diagram 3->4 (entering from left, col0 side) means action 9 (col4, right)?
    # action 9 = col4 right; kichwa_col(right=1)=col0. Sow 1 seed at col0 (was 0 -> empty -> done).
    s2 = g.step(s, 9)  # col4, right -> enter from left kichwa (col0)
    result = s2.board[2, ::-1]
    # Your front: col4=8, col0=1
    expected = jnp.array([1, 0, 0, 0, 8, 0, 2, 1], jnp.int16)
    assert jnp.array_equal(result, expected), f"got {result}"


def test_diagram14_to_15_chain_capture():
    """Diagram 14->15: col1 (left kimbi), chain capture."""
    s = make_state([
        [0, 2, 1, 0, 8, 0, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 4, 8, 0, 2, 5, 6],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    s2 = g.step(s, 3)  # col1, right (forced for left kimbi)
    result = s2.board[2, ::-1]
    expected = jnp.array([2, 5, 3, 1, 8, 0, 3, 4], jnp.int16)
    assert jnp.array_equal(result, expected), f"got {result}"


def test_diagram14_to_16_nyumba_continue():
    """Diagram 14->16: col7 (right kichwa), nyumba pending then continue."""
    s = make_state([
        [0, 2, 1, 0, 8, 0, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 4, 8, 0, 2, 5, 6],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    s2 = g.step(s, 14)  # col7, left (forced for right kichwa)
    assert bool(s2.nyumba_pending), "should be awaiting nyumba decision"
    assert int(s2.pending_direction) == 0, "direction should be left"

    s3 = g.step(s2, 33)  # continue
    front = s3.board[2, ::-1]
    back  = s3.board[3, ::-1]
    assert jnp.array_equal(front, jnp.array([1, 3, 3, 2, 0, 2, 5, 7], jnp.int16)), f"front {front}"
    assert jnp.array_equal(back,  jnp.array([1, 1, 1, 1, 1, 1, 0, 0], jnp.int16)), f"back {back}"


def test_diagram14_to_16_nyumba_stop():
    """Nyumba stop: nyumba seeds remain; it is now opponent's turn."""
    s = make_state([
        [0, 2, 1, 0, 8, 0, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 4, 8, 0, 2, 5, 6],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    s2 = g.step(s, 14)
    s3 = g.step(s2, 32)  # stop
    assert not bool(s3.nyumba_pending)
    # After stop+flip the nyumba (now in board[2] from new player's view) should have seeds
    # Our old nyumba was at board[0][4]; after flip it's board[2][3]
    assert s3.board[2, 3] > 0, "nyumba should still have seeds after stop"


def test_nyumba_action_mask_when_pending():
    """When nyumba_pending, only actions 32 and 33 are legal."""
    s = make_state([
        [0, 2, 1, 0, 8, 0, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 4, 8, 0, 2, 5, 6],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    s2 = g.step(s, 14)
    assert bool(s2.nyumba_pending)
    mask = g.legal_action_mask(s2)
    legal = set(jnp.where(mask)[0].tolist())
    assert legal == {32, 33}


def test_seed_conservation_namua():
    """Total seeds (board + stocks) must stay constant throughout namua."""
    s = g.init()
    total = int(s.board.sum()) + int(s.stock.sum())
    for _ in range(10):
        mask = g.legal_action_mask(s)
        action = jnp.where(mask)[0][0]
        s = g.step(s, action)
        if g.is_terminal(s):
            break
        new_total = int(s.board.sum()) + int(s.stock.sum())
        assert new_total == total, f"seed count changed: {new_total} != {total}"


def test_kichwa_direction_constraint():
    """Right kimbi capture should only allow leftward direction."""
    s = make_state([
        [0, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert 12 in legal,      "col6 left (action 12) should be legal"
    assert 13 not in legal,  "col6 right (action 13) should be illegal for right kimbi"


def test_nyumba_only_special_legal_mask():
    """When nyumba (>=6 seeds) is the only front hole and no capture exists,
    only actions 8 (nyumba-left) and 9 (nyumba-right) are legal."""
    s = make_state([
        [0, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], stock=(1, 0), nyumba_active=(True, True), stage=0)
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert legal == {8, 9}, f"expected {{8,9}}, got {legal}"


def test_nyumba_only_special_not_triggered_below_six():
    """When nyumba has < 6 seeds and is the only front hole, normal takasa applies
    (all seeds sowed, not the 2-seed special move)."""
    s = make_state([
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], stock=(1, 0), nyumba_active=(True, True), stage=0)
    # Normal takasa: both directions on col4 are legal
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    assert legal == {8, 9}, f"expected {{8,9}} (normal takasa), got {legal}"
    # After executing, all 6 seeds (5+1 stock) should have been sowed out, not just 2
    s2 = g.step(s, jnp.int32(9))  # right
    # Our nyumba is now board[2,3] from opponent's view; it should be empty (sowed out)
    assert s2.board[2, 3] == 0, f"nyumba should be empty after normal takasa, got {s2.board[2,3]}"


def test_nyumba_only_special_two_seed_execution():
    """Executing the special nyumba move places 1 stock seed in, takes 2 out,
    sows those 2 seeds, and leaves the nyumba with nyumba+1-2 seeds."""
    s = make_state([
        [0, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], stock=(1, 0), nyumba_active=(True, True), stage=0)
    s2 = g.step(s, jnp.int32(8))  # nyumba-left: sow 2 seeds leftward
    # After move: nyumba had 7, +1 stock, -2 sowed = 6 remaining.
    # After flip the nyumba (board[0,4]) becomes board[2,3] from opponent's view.
    assert s2.board[2, 3] == 6, f"nyumba should have 6 seeds, got {s2.board[2,3]}"
    # 2 seeds sowed left from nyumba: land at col3 (pos-1) and col2 (pos-2) from cur player view;
    # after flip those become board[2,4] and board[2,5].
    assert s2.board[2, 4] == 1, f"expected 1 at col4 of opp front, got {s2.board[2,4]}"
    assert s2.board[2, 5] == 1, f"expected 1 at col5 of opp front, got {s2.board[2,5]}"


def test_mtaji_moja():
    """Diagram 24: opponent is in a takasa situation.

    Board (from current player's view, opponent to move):
      opp back:   0 0 0 0 0 0 0 0
      opp front:  0 0 0 0 5 6 0 0   ← 5 seeds at col4, 6 seeds at col5
      cur front:  0 3 0 0 4 1 0 0   ← 4 seeds at col4 (only non-singleton mtaji), 1 at col5
      cur back:   0 7 0 0 0 0 0 0

    After flipping to opponent's turn:
      board[0] = [0,0,6,5,0,0,0,0]  (their front: 6 at col2, 5 at col3)
      board[2] = [0,0,1,4,0,0,3,0]  (our front reversed: col2=1 singleton, col3=4, col6=3)

    Paired non-singleton holes (board[0,c]>0 AND board[2,c]>=2):
      only col3 qualifies (col2 is a singleton → excluded).
    → mtaji-moja active at col3: the 5-seed hole (col3) is blocked.
    → only the 6-seed hole (col2) may be sowed, in either direction.
    """
    # Build the GameState from the opponent's perspective (after the board flip).
    s = make_state(
        [
            [0, 0, 6, 5, 0, 0, 0, 0],  # opp front (their view)
            [0, 0, 0, 0, 0, 0, 0, 0],  # opp back
            [0, 0, 1, 4, 0, 0, 3, 0],  # cur front (reversed)
            [0, 0, 0, 0, 0, 0, 7, 0],  # cur back (reversed)
        ],
        stock=(0, 0),
        nyumba_active=(False, False),
        stage=1,
    )
    mask = g.legal_action_mask(s)
    legal = set(jnp.where(mask)[0].tolist())
    # col2 (6 seeds): both directions legal
    assert 4 in legal, "col2-left (6-seed hole) must be legal"
    assert 5 in legal, "col2-right (6-seed hole) must be legal"
    # col3 (5 seeds): blocked by moja (our 4-seed hole at col3 is the sole mtaji)
    assert 6 not in legal, "col3-left (5-seed hole) must be blocked by moja"
    assert 7 not in legal, "col3-right (5-seed hole) must be blocked by moja"


def test_a1r_capture_chain_no_nyumba_pending():
    """Namua capture at A1 (left kichwa) with a 3-seed nyumba.

    The 1 captured seed re-enters from kichwa col0, picks up the col0 pile,
    sows rightward and reaches the nyumba.  The nyumba would receive its 4th
    seed there — but the stop/continue rule only applies when the nyumba holds
    >= 6 seeds (RULES §The Nyumba), so the sow continues normally past it,
    wraps around through the back row, and produces a chain of captures.

    Initial position (P0 perspective):
      A: [2,10, 1, 0, 3, 1, 0, 0]  B: [0, 3, 3, 1, 2, 1, 1, 1]
      a: [1, 0, 2, 2, 0, 2, 1, 0]  b: [2, 0, 2, 0, 2, 1, 3, 1]
      stock: [8, 8]
    """
    s = make_state(
        [
            [2, 10, 1, 0, 3, 1, 0, 0],
            [0,  3, 3, 1, 2, 1, 1, 1],
            [1,  0, 2, 2, 0, 2, 1, 0],
            [2,  0, 2, 0, 2, 1, 3, 1],
        ],
        stock=(8, 8),
        nyumba_active=(True, True),
        stage=0,
    )
    total_before = int(s.board.sum()) + int(s.stock.sum())

    # A1R = action 1 (col0, right).  col0 is left kichwa; opp has 1 seed at a1.
    s2 = g.step(s, jnp.int32(1))

    # Nyumba only reaches 4 seeds during the sow → no stop/continue option.
    assert not bool(s2.nyumba_pending), \
        "nyumba_pending must NOT trigger (nyumba has <6 seeds)"
    assert int(s2.current_player) == 1, "turn passes to P1 (no decision pending)"

    # Verify final board state, now from P1's perspective after the flip.
    assert jnp.array_equal(s2.board[0], jnp.array([0, 1, 2, 0, 2, 0, 0, 0], jnp.int16)), \
        f"P1 front: {s2.board[0]}"
    assert jnp.array_equal(s2.board[1], jnp.array([1, 3, 1, 2, 0, 2, 0, 2], jnp.int16)), \
        f"P1 back: {s2.board[1]}"
    assert jnp.array_equal(s2.board[2], jnp.array([2, 2, 3, 1, 2, 4, 0, 2], jnp.int16)), \
        f"P0 front (from P1 view): {s2.board[2]}"
    assert jnp.array_equal(s2.board[3], jnp.array([1, 3, 1, 4, 1, 5, 1, 1], jnp.int16)), \
        f"P0 back  (from P1 view): {s2.board[3]}"

    # Stock from P1's perspective: [P1's, P0's].  P0 used one stock seed.
    assert int(s2.stock[0]) == 8, "P1 stock unchanged"
    assert int(s2.stock[1]) == 7, "P0 stock decreased by 1"

    # P0's nyumba was emptied mid-sow (3 seeds picked up by do_continue) — even
    # though wraparound later placed 1 seed back, the nyumba must be deactivated.
    # After the flip nyumba_active = [P1, P0], so index 1 is P0's.
    assert not bool(s2.nyumba_active[1]), \
        f"P0 nyumba must deactivate when emptied mid-sow, got {s2.nyumba_active}"

    # Seed conservation across the whole chain.
    total_after = int(s2.board.sum()) + int(s2.stock.sum())
    assert total_before == total_after, f"seeds: {total_before} -> {total_after}"


def test_nyumba_deactivates_when_emptied_then_refilled():
    """RULES: 'Once the nyumba's seeds have been sown out, it becomes an
    ordinary hole.'  Even if a sow empties the nyumba and then the same chain
    later places seeds back into it, the nyumba must still be deactivated.

    Setup (mtaji): board[0,4]=2 (nyumba), board[2,0]=1, board[0,0]=0.  Source
    move col4-right (action 9): the 2 seeds reach col5, col6 — but col6=1
    (singleton chain).  Use a simpler construction below.
    """
    # Sow the nyumba directly: source pickup empties it, then a longer chain
    # wraps around and drops a seed back in.  board[0,4]=2 seeds, sowed right.
    # next_pos(0,4,right) = (0,5).  Place at col5 (1+1=2, was occupied) →
    # do_continue picks up 2 → continues until a seed lands back in col4.
    s = make_state(
        [
            [0, 0, 0, 0, 2, 1, 0, 0],   # nyumba=2, col5=1 (chain bait)
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],   # no opp seeds → no capture
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        stock=(0, 0),
        nyumba_active=(True, False),
        stage=1,
    )
    # Use mtaji action 9 (col4 right) — picks up the nyumba's 2 seeds itself.
    s2 = g.step(s, jnp.int32(9))
    # After flip nyumba_active[1] is P0's (current player at start) nyumba.
    assert not bool(s2.nyumba_active[1]), \
        f"nyumba must deactivate after being sowed out, got {s2.nyumba_active}"


def test_nyumba_pending_only_when_active():
    """When nyumba_active[0]=False, the nyumba hole is an ordinary hole regardless
    of how many seeds it contains.  Landing the last seed there must NOT trigger
    nyumba_pending.

    Setup: board[0,4]=8 (nyumba hole with >=6 seeds — would trigger pending if
    active).  Action col3-right (action 7): sow 3 seeds; last seed lands at col4
    making it 9 seeds.  With nyumba_active=False the sow falls through to
    do_continue rather than pausing for a stop/continue decision.
    """
    s = make_state(
        [
            [0, 0, 0, 3, 8, 0, 0, 0],  # col3=3, col4=8 (>=6 seeds; would pend if active)
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        stock=(0, 0),
        nyumba_active=(False, False),
        stage=1,
    )
    # action 7 = col3 right (mtaji takasa, 3 seeds, last lands at col4)
    s2 = g.step(s, jnp.int32(7))
    assert not bool(s2.nyumba_pending), \
        "nyumba_pending must NOT be set when nyumba is inactive (even with >=6 seeds)"


def test_nyumba_pending_requires_six_seeds():
    """Landing the last seed at the nyumba while it holds <6 seeds must not
    trigger the stop/continue decision (RULES: nyumba special rules apply only
    while it contains 6 or more seeds).
    """
    # Active nyumba but only 4 seeds after the last seed is placed (3+1).
    s = make_state(
        [
            [0, 0, 0, 3, 3, 0, 0, 0],  # col3=3 source, col4=3 nyumba
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],  # no opp seeds anywhere
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        stock=(0, 0),
        nyumba_active=(True, True),
        stage=1,
    )
    s2 = g.step(s, jnp.int32(7))  # col3 right; 3 seeds land at col4 (1 seed left)
    assert not bool(s2.nyumba_pending), \
        "nyumba_pending must NOT trigger when the nyumba has <6 seeds"


def test_nyumba_as_takasa_source_does_not_repend_on_wraparound():
    """When the player uses the nyumba as a normal takasa source, the nyumba's
    seeds are sown out — it must be treated as an ordinary hole for the rest
    of the move.  If sowing wraps around and lands the last seed back at the
    nyumba (>=6 seeds), pending must NOT trigger.

    Setup: nyumba has 7 seeds, every other front/back hole has 1 seed.  Source
    = nyumba, direction = right.  Stock seed makes 8 to sow.  Path: col5..col7,
    back col7..col0, front col0..col3 (8 seeds total).  Last seed lands at
    col3 with board[0,3]=1+1=2 (occupied), board[2,3]=0 -> do_continue, not
    pending.  No nyumba_pending fires; turn ends after chain.
    """
    s = make_state(
        [
            [1, 1, 1, 1, 7, 1, 1, 1],   # nyumba=7 + 1s elsewhere
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],   # no opp seeds -> takasa
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        stock=(1, 0),
        nyumba_active=(True, True),
        stage=0,
    )
    # action 9 = col4 right (takasa source = nyumba)
    s2 = g.step(s, jnp.int32(9))
    assert not bool(s2.nyumba_pending), \
        f"nyumba sowed-out must not re-pend on wraparound, got pending={s2.nyumba_pending}"
    # And the nyumba must be deactivated (sowed out by source pickup).
    assert not bool(s2.nyumba_active[1]), \
        f"nyumba must deactivate after source pickup, got {s2.nyumba_active}"


def test_opp_nyumba_deactivated_on_namua_capture():
    """Opponent nyumba at board[2,3] should be deactivated when captured in namua."""
    # col3 has 1 seed on our side and 7 on opp side (their active nyumba).
    # Stock capture adds 1 to col3, takes 7 from opp, sows from kichwa.
    s = make_state([
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], stock=(1, 0), nyumba_active=(False, True), stage=0)
    # action 7 = col3 right (col3 is not kichwa/kimbi, both dirs legal)
    s2 = g.step(s, jnp.int32(7))
    assert jnp.array_equal(s2.nyumba_active, jnp.array([False, False])), \
        f"nyumba_active={s2.nyumba_active}, expected [False, False]"


def test_opp_nyumba_deactivated_on_mtaji_capture():
    """Opponent nyumba at board[2,3] should be deactivated when captured in mtaji.

    Setup: board[0,0]=3 (left kichwa, 3 seeds).  board[0,3]=1 (occupied).
    board[2,3]=7 (opponent nyumba).  Action 1 = col0 right (forced direction).
    Sowing 3 seeds right from col0 lands last seed at col3; col3 was occupied
    so capture fires, clearing board[2,3].
    """
    s = make_state([
        [3, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], stock=(0, 0), nyumba_active=(False, True), stage=1)
    # action 1 = col0 right (only legal direction for left kichwa)
    s2 = g.step(s, jnp.int32(1))
    assert jnp.array_equal(s2.nyumba_active, jnp.array([False, False])), \
        f"nyumba_active={s2.nyumba_active}, expected [False, False]"
