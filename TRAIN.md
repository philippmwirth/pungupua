# Training Loop Design — AlphaZero for Bao la Kiswahili

AlphaZero-style self-play training using `mctx` (Gumbel MuZero policy) on
top of the existing `Bao` pgx environment.  The design targets a single
M2 MacBook for local testing and is written to scale to multi-device later
with minimal changes.

---

## File layout

```
train/
├── network.py      # Haiku network: conv backbone + policy/value heads
├── config.py       # Pydantic config with M2-friendly defaults
└── train.py        # Main script: selfplay → loss → update loop
```

---

## Dependencies (to add to pyproject.toml)

```
mctx          # MCTS library (Gumbel MuZero policy)
dm-haiku      # Neural network library
optax         # Gradient-based optimisers
wandb         # Experiment tracking (optional; gated by config.use_wandb)
omegaconf     # CLI config overrides
pydantic      # Config validation
```

---

## Network — `train/network.py`

The observation tensor has shape `(4, 8, 67)`:
rows × cols × channels (65 one-hot seed counts + 2 nyumba flags).
We treat this as a spatial image with 67 input channels.

```
Input: (batch, 4, 8, 67)
  │
  ▼
Conv 3×3, C channels, same padding   ← projects to C channels
  │
  ▼
N × ResBlock(C channels)             ← each block: BN→ReLU→Conv→BN→ReLU→Conv + skip
  │
  ├──► Policy head
  │      GlobalAvgPool → Dense(C) → ReLU → Dense(num_actions=34)
  │      (raw logits; softmax / masking done outside the network)
  │
  └──► Value head
         GlobalAvgPool → Dense(C) → ReLU → Dense(1) → tanh
         (scalar in [-1, 1])
```

Implementation uses `hk.transform_with_state` (for batch-norm running stats).
`forward_fn(x, is_eval=False)` returns `(logits, value)` where value is
squeezed to shape `(batch,)`.

---

## Config — `train/config.py`

```python
class Config(BaseModel):
    seed: int = 0
    max_num_iters: int = 200

    # network
    num_channels: int = 64       # M2: 64; scale → 128+
    num_blocks: int = 4          # M2: 4;  scale → 8+

    # self-play
    selfplay_batch_size: int = 32   # M2: 32; scale → 512+
    num_simulations: int = 16       # M2: 16; scale → 64+
    max_num_steps: int = 200        # hard cap per episode scan

    # training
    training_batch_size: int = 128  # M2: 128; scale → 1024+
    learning_rate: float = 1e-3

    # eval
    eval_interval: int = 10
    use_wandb: bool = False         # off by default for local runs
```

`OmegaConf.from_cli()` lets you override any field:
`python train/train.py num_simulations=32 use_wandb=true`

---

## No `pmap` on M2

The reference uses `@jax.pmap` to shard across multiple accelerators.
On M2 there is exactly one CPU device; pmap works but adds overhead and
complexity.  We use `@jax.jit` instead throughout.

When scaling later, switching to pmap requires:
1. Replace `@jax.jit` with `@partial(jax.pmap, axis_name="i")` on
   `selfplay`, `compute_loss_input`, `train`.
2. Average gradients with `jax.lax.pmean(..., axis_name="i")` in `train`.
3. Split keys per device before each call.
4. `jax.device_put_replicated((model, opt_state), devices)` at init.

---

## `recurrent_fn`

Identical in structure to the reference but no changes needed for Bao —
the pgx `Bao.step` already handles the current-player perspective flip,
nyumba_pending decisions, and stage transitions transparently.

```python
def recurrent_fn(model, rng_key, action, state):
    model_params, model_state = model
    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True
    )
    logits -= jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value  = jnp.where(state.terminated, 0.0, value)
    discount = jnp.where(state.terminated, 0.0, -jnp.ones_like(value))

    return mctx.RecurrentFnOutput(
        reward=reward, discount=discount,
        prior_logits=logits, value=value,
    ), state
```

The `-1` discount encodes the zero-sum, alternating-player structure: my
value in the next state is the negative of the opponent's value.

---

## `selfplay` — `@jax.jit`

```
keys = split(rng_key, selfplay_batch_size)
state = vmap(env.init)(keys)       ← batch of fresh games

jax.lax.scan over max_num_steps:
  for each step:
    1. forward(state.observation)   → root logits + value
    2. mctx.gumbel_muzero_policy    → action_weights + action
    3. vmap(auto_reset(env.step))(state, action, keys)
    4. yield SelfplayOutput(obs, action_weights, reward, terminated, discount)
```

`auto_reset` restarts a finished game immediately so the batch stays full
and `jax.lax.scan` can use static shapes.

---

## `compute_loss_input` — `@jax.jit`

Converts raw selfplay output into training targets:

- **`value_tgt`**: discounted return computed backwards via `jax.lax.scan`:
  `v[t] = reward[t] + discount[t] * v[t+1]`
  with `v[T] = 0`.  The alternating `-1` discount propagates win/loss
  backwards through the game tree.
- **`mask`**: `cumsum(terminated[::-1])[::-1] >= 1` — True for every
  timestep that is part of a completed episode.  Truncated tails (games
  still running at `max_num_steps`) are masked out of the value loss.

---

## `loss_fn`

```python
def loss_fn(params, net_state, samples):
    (logits, value), net_state = forward.apply(params, net_state, samples.obs)

    policy_loss = mean(softmax_cross_entropy(logits, samples.policy_tgt))
    value_loss  = mean(l2_loss(value, samples.value_tgt) * samples.mask)

    return policy_loss + value_loss, (net_state, policy_loss, value_loss)
```

---

## `train` step — `@jax.jit`

```python
grads, (net_state, pl, vl) = jax.grad(loss_fn, has_aux=True)(params, net_state, batch)
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

No `pmean` needed on a single device.

---

## Training loop (pseudocode)

```
init model, opt_state, rng_key

for iteration in range(max_num_iters):

    if iteration % eval_interval == 0:
        evaluate()     ← greedy play vs random baseline
        checkpoint()   ← pickle to checkpoints/

    # --- self-play ---
    data    = selfplay(model, rng_key)          # SelfplayOutput
    samples = compute_loss_input(data)          # Sample

    # flatten (max_num_steps × batch) → N, shuffle, make minibatches
    samples = flatten_and_shuffle(samples, rng_key)
    minibatches = batch_into(samples, training_batch_size)

    # --- training ---
    for mb in minibatches:
        model, opt_state, pl, vl = train(model, opt_state, mb)

    log(iteration, pl, vl, frames, hours)
```

---

## Evaluation

Simple greedy evaluation (no MCTS, no baseline model):
- Player 0 uses the current network (argmax of logits).
- Player 1 plays uniformly at random from legal actions.
- Run `selfplay_batch_size` games, report win/draw/loss rate.

A proper evaluation (MCTS vs MCTS, or tournament against a prior checkpoint)
is left for later; the greedy proxy is sufficient to verify learning.

---

## Bao-specific notes

### Observation shape
`(4, 8, 67)` — handled by the conv network directly (rows=4, cols=8).

### Action space
34 actions, 6–12 legal at most positions.  `legal_action_mask` is always
attached to `state`; masking in `recurrent_fn` sets invalid logits to `-inf`.

### No truncation
Bao has no time limit in the rules, but `jax.lax.scan` requires a fixed
number of steps.  We set `max_num_steps=200` as a soft cap and mask out
unfinished episodes from the value loss.  If cycles become a problem in
self-play (games that never terminate), a step-count limit can be added
to `Bao._step`.

### Discount
Always `-1.0` for non-terminal steps.  This is correct for any two-player
zero-sum game where turns alternate: if the current player wins next turn,
that is a loss for the player who just moved.

---

## Scaling checklist (M2 → multi-device)

1. Add `num_devices = len(jax.local_devices())` and `batch_size // num_devices`.
2. Wrap `selfplay`, `compute_loss_input`, `train` with `jax.pmap`.
3. Add `jax.lax.pmean(grads, axis_name="i")` in `train`.
4. `jax.device_put_replicated((model, opt_state), devices)` at init.
5. Split rng keys per device before each pmap call.
6. Increase `num_channels`, `num_blocks`, `selfplay_batch_size`, `num_simulations`.
