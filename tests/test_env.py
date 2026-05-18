"""Tests for the pgx wrapper (`bao.env`).

Covers:
  - State defaults and INIT_LEGAL_ACTION_MASK contents.
  - Bao env properties (id, version, num_players).
  - init: shapes/dtypes, current_player/_player_order consistency, randomization.
  - step: turn alternation, nyumba_pending keeps the player to move,
    terminal handling, reward reordering by _player_order.
  - observe: current-player vs opponent perspective.
  - vmap support: batched init and step.
  - Illegal action: pgx framework terminates with a negative reward.
"""

import jax
import jax.numpy as jnp

from bao import (
    INIT_LEGAL_ACTION_MASK,
    NUM_ACTIONS,
    NYUMBA_CONTINUE,
    NYUMBA_STOP,
    Bao,
    Game,
    GameState,
    State,
)


# ---------------------------------------------------------------------------
# INIT_LEGAL_ACTION_MASK and State defaults
# ---------------------------------------------------------------------------


def test_init_legal_action_mask_shape_and_contents():
    """The precomputed init mask has bits set exactly at {8..13}: both
    directions for the three occupied front holes (cols 4, 5, 6) in the
    canonical opening (no capture is possible from the symmetric start)."""
    assert INIT_LEGAL_ACTION_MASK.shape == (NUM_ACTIONS,)
    assert INIT_LEGAL_ACTION_MASK.dtype == jnp.bool_
    legal = set(jnp.where(INIT_LEGAL_ACTION_MASK)[0].tolist())
    assert legal == {8, 9, 10, 11, 12, 13}, f"got {legal}"


def test_init_legal_action_mask_matches_game_init():
    """INIT_LEGAL_ACTION_MASK must equal what Game.legal_action_mask returns
    for Game.init() — this is the contract used as a dataclass default."""
    g = Game()
    expected = g.legal_action_mask(g.init())
    assert jnp.array_equal(INIT_LEGAL_ACTION_MASK, expected)


def test_state_defaults_have_correct_shapes_and_dtypes():
    """Constructing a default State (no args) must give the canonical types
    that pgx algorithms rely on when creating zero/template states."""
    s = State()
    assert s.current_player.shape == () and s.current_player.dtype == jnp.int32
    assert s.rewards.shape == (2,) and s.rewards.dtype == jnp.float32
    assert s.terminated.shape == () and s.terminated.dtype == jnp.bool_
    assert s.truncated.shape == () and s.truncated.dtype == jnp.bool_
    assert s.legal_action_mask.shape == (NUM_ACTIONS,)
    assert s.legal_action_mask.dtype == jnp.bool_
    assert s.observation.shape == (4, 8, 67)
    assert s.observation.dtype == jnp.float32
    assert s._player_order.shape == (2,) and s._player_order.dtype == jnp.int32
    assert isinstance(s._x, GameState)


def test_state_env_id_is_bao():
    assert State().env_id == "bao"


# ---------------------------------------------------------------------------
# Bao env: static metadata
# ---------------------------------------------------------------------------


def test_bao_env_metadata():
    env = Bao()
    assert env.id == "bao"
    assert env.version == "v0"
    assert env.num_players == 2


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_returns_consistent_state():
    """env.init must produce a State whose fields agree with the underlying
    GameState: current_player matches _player_order[x.current_player], the
    legal mask matches Game.legal_action_mask(x), terminated/truncated are
    False, rewards are zero, observation has the expected shape."""
    env = Bao()
    g = Game()
    state = env.init(jax.random.PRNGKey(0))

    assert isinstance(state, State)
    assert state.terminated == jnp.bool_(False)
    assert state.truncated == jnp.bool_(False)
    assert jnp.array_equal(state.rewards, jnp.zeros(2, jnp.float32))

    # current_player is the pgx id of whoever the internal state says is to move.
    assert int(state.current_player) == int(
        state._player_order[state._x.current_player]
    )

    # Mask matches what Game would return for this internal state.
    assert jnp.array_equal(state.legal_action_mask, g.legal_action_mask(state._x))

    # Observation shape comes from pgx's init wrapper (fills via _observe).
    assert state.observation.shape == (4, 8, 67)
    assert state.observation.dtype == jnp.float32


def test_init_observation_matches_current_player_view():
    """env.init's observation is taken from current_player's perspective.  At
    the symmetric opening, both perspectives are identical."""
    env = Bao()
    g = Game()
    state = env.init(jax.random.PRNGKey(0))
    expected = g.observe(state._x, state._player_order[state.current_player])
    assert jnp.allclose(state.observation, expected)


def test_init_player_order_randomization():
    """Across many keys, both [0,1] and [1,0] orderings appear with some
    frequency (the Bernoulli is unbiased)."""
    env = Bao()
    seen_first_internal = set()
    for seed in range(32):
        state = env.init(jax.random.PRNGKey(seed))
        seen_first_internal.add(tuple(state._player_order.tolist()))
    assert seen_first_internal == {(0, 1), (1, 0)}, f"got {seen_first_internal}"


def test_init_current_player_consistent_with_player_order():
    """For every key, current_player[init] equals _player_order[0] — since
    Game.init starts with internal current_player = 0."""
    env = Bao()
    for seed in range(8):
        state = env.init(jax.random.PRNGKey(seed))
        assert int(state.current_player) == int(state._player_order[0])
        assert int(state._x.current_player) == 0


# ---------------------------------------------------------------------------
# step: turn alternation and nyumba_pending
# ---------------------------------------------------------------------------


def test_step_alternates_current_player_on_normal_move():
    """A non-pending namua move flips current_player between the two pgx ids."""
    env = Bao()
    state = env.init(jax.random.PRNGKey(0))
    before = int(state.current_player)
    # action 8 = col4 left, legal at the opening, not a capture, no pending.
    state2 = env.step(state, jnp.int32(8))
    after = int(state2.current_player)
    assert after == 1 - before, f"player did not alternate ({before} -> {after})"
    assert not bool(
        state2.nyumba_pending
        if hasattr(state2, "nyumba_pending")
        else state2._x.nyumba_pending
    )


def test_step_preserves_current_player_during_nyumba_pending():
    """When a move triggers nyumba_pending, the same pgx player must move
    again (to choose stop/continue) — current_player does not flip."""
    env = Bao()
    g = Game()

    # Construct a state where the next move triggers nyumba_pending.  Using
    # diagram 14 -> 16: action 14 (col7 left, right kichwa forced) ends with
    # the last seed in the nyumba.
    x = GameState(
        board=jnp.array(
            [
                [0, 2, 1, 0, 8, 0, 3, 4],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 4, 8, 0, 2, 5, 6],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            jnp.int16,
        ),
        stock=jnp.array([10, 10], jnp.int16),
        current_player=jnp.int32(0),
        stage=jnp.int32(0),
        winner=jnp.int32(-1),
        nyumba_active=jnp.array([True, True]),
        nyumba_pending=jnp.bool_(False),
        pending_direction=jnp.int32(1),
    )
    state = State(  # type: ignore[call-arg]
        current_player=jnp.int32(0),
        legal_action_mask=g.legal_action_mask(x),
        _player_order=jnp.array([0, 1], jnp.int32),
        _x=x,
    )
    before = int(state.current_player)
    state2 = env.step(state, jnp.int32(14))
    assert bool(state2._x.nyumba_pending), "expected nyumba_pending"
    assert int(state2.current_player) == before, (
        "current_player must not flip while nyumba_pending"
    )
    # And the only legal actions are stop/continue.
    legal = set(jnp.where(state2.legal_action_mask)[0].tolist())
    assert legal == {NYUMBA_STOP, NYUMBA_CONTINUE}


# ---------------------------------------------------------------------------
# step: terminal + reward reordering
# ---------------------------------------------------------------------------


def _terminal_state_with_player_order(env: Bao, order: tuple[int, int]) -> State:
    """Build a state that becomes terminal on action 1 (col0 right capture),
    parameterized by the pgx _player_order so we can verify reward indexing.

    Setup: cur has 3 at col0 (left kichwa, forced right), 1 at col3.  Opp has
    2 at col3.  Sowing 3 seeds from col0 right lands the last at col3,
    capturing the 2 -> opp front cleared -> winner = internal current_player.
    """
    g = env.game
    x = GameState(
        board=jnp.array(
            [
                [3, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            jnp.int16,
        ),
        stock=jnp.array([0, 0], jnp.int16),
        current_player=jnp.int32(0),
        stage=jnp.int32(1),
        winner=jnp.int32(-1),
        nyumba_active=jnp.array([False, False]),
        nyumba_pending=jnp.bool_(False),
        pending_direction=jnp.int32(1),
    )
    po = jnp.array(order, jnp.int32)
    return State(  # type: ignore[call-arg]
        current_player=po[x.current_player],
        legal_action_mask=g.legal_action_mask(x),
        _player_order=po,
        _x=x,
    )


def test_step_terminal_rewards_identity_player_order():
    """With _player_order = [0, 1] and pgx_0 winning, rewards = [+1, -1]."""
    env = Bao()
    state = _terminal_state_with_player_order(env, (0, 1))
    state2 = env.step(state, jnp.int32(1))
    assert bool(state2.terminated)
    assert jnp.allclose(state2.rewards, jnp.array([1.0, -1.0], jnp.float32))


def test_step_terminal_rewards_swapped_player_order():
    """With _player_order = [1, 0], the winning internal player 0 is pgx_1,
    so rewards must be [-1, +1] (indexed by pgx id)."""
    env = Bao()
    state = _terminal_state_with_player_order(env, (1, 0))
    state2 = env.step(state, jnp.int32(1))
    assert bool(state2.terminated)
    assert jnp.allclose(state2.rewards, jnp.array([-1.0, 1.0], jnp.float32))


# ---------------------------------------------------------------------------
# observe: per-player perspective
# ---------------------------------------------------------------------------


def test_observe_current_player_matches_default_observe():
    """observe(state, current_player) must equal Game.observe(_x, internal color)."""
    env = Bao()
    g = Game()
    state = env.init(jax.random.PRNGKey(3))
    obs = env.observe(state, state.current_player)
    expected = g.observe(state._x, state._player_order[state.current_player])
    assert jnp.allclose(obs, expected)


def test_observe_opponent_view_is_flipped():
    """observe(state, opponent) shows the board from the opposite perspective.

    Build an asymmetric position so the two perspectives are distinguishable.
    """
    env = Bao()
    g = Game()
    x = GameState(
        board=jnp.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 9],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            jnp.int16,
        ),
        stock=jnp.array([5, 5], jnp.int16),
        current_player=jnp.int32(0),
        stage=jnp.int32(0),
        winner=jnp.int32(-1),
        nyumba_active=jnp.array([True, True]),
        nyumba_pending=jnp.bool_(False),
        pending_direction=jnp.int32(1),
    )
    state = State(  # type: ignore[call-arg]
        current_player=jnp.int32(0),
        legal_action_mask=g.legal_action_mask(x),
        _player_order=jnp.array([0, 1], jnp.int32),
        _x=x,
    )
    obs_cur = env.observe(state, jnp.int32(0))
    obs_opp = env.observe(state, jnp.int32(1))
    assert not jnp.allclose(obs_cur, obs_opp), (
        "asymmetric position must yield distinct views"
    )
    # And opp view equals Game.observe with color = the opponent's internal id.
    expected_opp = g.observe(state._x, jnp.int32(1))
    assert jnp.allclose(obs_opp, expected_opp)


# ---------------------------------------------------------------------------
# vmap support
# ---------------------------------------------------------------------------


def test_vmap_init_produces_batched_state():
    """jax.vmap(env.init) over a batch of keys produces a batched State."""
    env = Bao()
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    states = jax.vmap(env.init)(keys)
    assert states.current_player.shape == (16,)
    assert states._player_order.shape == (16, 2)
    assert states.observation.shape == (16, 4, 8, 67)
    assert states.legal_action_mask.shape == (16, NUM_ACTIONS)
    # All lanes start non-terminal.
    assert not bool(states.terminated.any())
    # And the Bernoulli must have flipped some lanes (extremely unlikely otherwise).
    orderings = {tuple(po.tolist()) for po in states._player_order}
    assert orderings == {(0, 1), (1, 0)}


def test_vmap_step_runs_independently_per_lane():
    """A vmapped step must produce a per-lane next state without crosstalk."""
    env = Bao()
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    states = jax.vmap(env.init)(keys)
    actions = jnp.array([8, 9, 10, 11], jnp.int32)
    nexts = jax.vmap(env.step)(states, actions)
    # current_player should have flipped in each lane (no nyumba_pending at init).
    flipped = (nexts.current_player == (1 - states.current_player)).all()
    assert bool(flipped)


# ---------------------------------------------------------------------------
# Illegal action: pgx terminates with negative reward for the offender
# ---------------------------------------------------------------------------


def test_step_with_illegal_action_terminates_with_loss_for_acting_player():
    """pgx's Env.step penalises taking an illegal action by terminating
    immediately with -1 for the player who acted (and +1 for the other)."""
    env = Bao()
    state = env.init(jax.random.PRNGKey(0))
    # action 0 (col0 left) is not legal at the symmetric opening.
    assert not bool(state.legal_action_mask[0])
    acting = int(state.current_player)
    state2 = env.step(state, jnp.int32(0))
    assert bool(state2.terminated), "illegal action must terminate the game"
    assert float(state2.rewards[acting]) == -1.0
    assert float(state2.rewards[1 - acting]) == 1.0
