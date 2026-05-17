"""Property tests: seed conservation, no illegal captures, terminal detection."""

import jax
import jax.numpy as jnp
from bao import Game
from bao.state import GameState


g = Game()


def _total_seeds(s: GameState) -> int:
    return int(s.board.sum()) + int(s.stock.sum())


def _play_game(rng, max_steps=500):
    """Play a random game, returning (states, final_state)."""
    s = g.init()
    states = [s]
    for _ in range(max_steps):
        if g.is_terminal(s):
            break
        mask = g.legal_action_mask(s)
        legal = jnp.where(mask)[0]
        rng, sub = jax.random.split(rng)
        action = legal[jax.random.randint(sub, (), 0, len(legal))]
        s = g.step(s, action)
        states.append(s)
    return states, s


def test_seed_conservation_full_game():
    """Seeds are never created or destroyed during a full game."""
    rng = jax.random.PRNGKey(0)
    states, _ = _play_game(rng)
    base = _total_seeds(states[0])
    for i, s in enumerate(states):
        t = _total_seeds(s)
        assert t == base, f"step {i}: seed count {t} != {base}"


def test_game_terminates():
    """A randomly-played game terminates within 500 steps."""
    rng = jax.random.PRNGKey(42)
    _, final = _play_game(rng)
    assert g.is_terminal(final), "game did not terminate within 500 steps"
    assert int(final.winner) in (0, 1)


def test_rewards_consistent_with_winner():
    rng = jax.random.PRNGKey(7)
    _, final = _play_game(rng)
    if g.is_terminal(final):
        r = g.rewards(final)
        w = int(final.winner)
        assert float(r[w]) == 1.0
        assert float(r[1 - w]) == -1.0


def test_legal_mask_always_nonempty_nonterminal():
    """Non-terminal states always have at least one legal action."""
    rng = jax.random.PRNGKey(13)
    states, _ = _play_game(rng)
    for s in states:
        if not g.is_terminal(s):
            mask = g.legal_action_mask(s)
            assert mask.any(), "non-terminal state has no legal actions"


def test_no_singletons_sowed_in_mtaji():
    """In mtaji, holes with exactly 1 seed must not appear as legal sow sources.

    Use two opponent holes so mtaji-moja does not block the col3 takasa source.
    """
    board = jnp.array(
        [
            [
                0,
                1,
                2,
                3,
                0,
                0,
                0,
                0,
            ],  # col2=2 added so both col2 and col3 are paired non-singletons
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0, 0, 0],  # two paired non-singleton holes -> no mtaji-moja
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        jnp.int16,
    )
    s = GameState(
        board=board,
        stock=jnp.array([0, 0], jnp.int16),
        current_player=jnp.int32(0),
        stage=jnp.int32(1),
        winner=jnp.int32(-1),
        nyumba_active=jnp.array([False, False]),
        nyumba_pending=jnp.bool_(False),
        pending_direction=jnp.int32(1),
    )
    mask = g.legal_action_mask(s)
    # col1 singleton must be blocked
    assert not mask[2], "col1 left should be illegal (singleton)"
    assert not mask[3], "col1 right should be illegal (singleton)"
    # col3 (3 seeds, no mtaji-moja) must be available for takasa
    assert mask[6] or mask[7], "col3 should have at least one legal takasa action"


def test_observe_shape():
    s = g.init()
    obs = g.observe(s)
    assert obs.shape == (4, 8, 67)
    assert obs.dtype == jnp.float32


def test_observe_symmetric_start():
    """Both players see the same observation from the symmetric starting position."""
    s = g.init()
    obs0 = g.observe(s, jnp.int32(0))
    obs1 = g.observe(s, jnp.int32(1))
    assert jnp.allclose(obs0, obs1)


def test_multiple_games_both_players_can_win():
    """Both players win at least once across several random games."""
    results = set()
    for seed in range(8):
        rng = jax.random.PRNGKey(seed)
        _, final = _play_game(rng)
        if g.is_terminal(final):
            results.add(int(final.winner))
    assert len(results) == 2, f"only player(s) {results} won across 8 games"
