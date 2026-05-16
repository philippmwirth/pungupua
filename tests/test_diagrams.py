"""Unit tests reproducing diagrams from RULES.md."""
import jax.numpy as jnp
import pytest
from bao import Game
from bao.state import GameState


def make_state(board, stock=(10, 10), nyumba_active=(True, True)):
    return GameState(
        board=jnp.array(board, jnp.int16),
        stock=jnp.array(stock, jnp.int16),
        current_player=jnp.int32(0),
        stage=jnp.int32(0),
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
