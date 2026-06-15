# Bug: `sow()` did not update direction on kichwa/kimbi mid-chain captures

**Status: FIXED** in `src/bao/sowing.py`.

---

## The rule

Multiple independent rule sources agree:

> *"Capturing in the flanks (kichwa/kimbi) is the **only way to change sowing
> direction in Bao within a move**."*
> — navpil (Medium); confirmed by gambiter.com

When a mid-chain capture lands at a kichwa or kimbi hole (cols 0, 1, 6, 7):
- The re-entry is forced to the **nearest kichwa** (same side as the capture).
- This may **flip the sowing direction**.

When a mid-chain capture lands elsewhere (cols 2–5): the current direction is
preserved and re-entry is from the kichwa of that direction (unchanged).

The initial-move kichwa/kimbi forcing (already implemented in `namua_step` and
the legal-action mask) is just the first application of this same rule.

---

## What was wrong

`sow()` carried `direction` as an immutable field throughout the while-loop.
`do_capture` always computed `kichwa_col(d_)` — using only the *current*
direction, ignoring whether the capture column was a kichwa/kimbi:

```python
# BEFORE (wrong)
def do_capture(args):
    b_, d_ = args
    captured = b_[2, c].astype(jnp.int16)
    b_ = b_.at[2, c].set(jnp.int16(0))
    kc = kichwa_col(d_)          # ← always current direction, never changes
    return b_, captured, jnp.int32(0), kc
```

**Concrete example of the error:**  
Sowing rightward (`d=1`), mid-chain capture at col 7 (right kichwa):
- **Was**: re-enter from col 0 (left kichwa), continue right. ✗
- **Should be**: re-enter from col 7 (right kichwa), continue left. ✓

---

## The fix

`do_capture` now recomputes direction from the capture column:

```python
# AFTER (correct)
def do_capture(args):
    b_, d_ = args
    captured = b_[2, c].astype(jnp.int16)
    b_ = b_.at[2, c].set(jnp.int16(0))
    new_d = jnp.where(is_kichwa_or_kimbi(c), forced_direction(c), d_)
    kc = kichwa_col(new_d)
    return b_, captured, jnp.int32(0), kc, new_d
```

`new_d` is now threaded through `do_continue`, `do_done`, `on_last`,
`on_not_last`, and propagated into the carry so all subsequent loop iterations
see the updated direction. The final direction is returned as `pending_dir`,
so `pending_direction` in the game state is also correct after nyumba
stop/continue decisions.

---

## Note on RULES.md

The original RULES.md contained the oversimplification:

> *"Once you have started sowing in a direction during a turn, you must
> continue in that same direction for subsequent sowing in that turn."*

This has been corrected to document the kichwa/kimbi exception.
