# PLAY.md — Hugging Face Space Design

Interactive web app for playing Bao la Kiswahili against trained AlphaZero bots, hosted as a Hugging Face Space.

---

## Overview

The Space lets visitors play Bao against any bot listed in `bots.jsonl`. Each bot entry specifies a network checkpoint, model config, and MCTS eval config. The human is always Player 0 (bottom side of the board).

**Stack**: Gradio (Python SDK) · JAX CPU · dm-haiku · mctx

---

## File Structure

```
pungupua/                        (existing repo)
├── app.py                       ← Gradio entry point (new)
├── space_requirements.txt       ← HF Space deps (new)
├── bots.jsonl                   ← bot registry (existing)
├── checkpoints/                 ← model weights (existing, uploaded to HF)
└── src/                         ← game engine (existing)
```

HF Space README front-matter (replaces current `README.md` header):
```yaml
---
title: Pungupua — Play Bao la Kiswahili
emoji: 🌰
colorFrom: amber
colorTo: green
sdk: gradio
sdk_version: "5.x"
app_file: app.py
pinned: false
---
```

---

## Dependencies

`space_requirements.txt`:
```
gradio>=5.0.0
jax[cpu]
jaxlib
dm-haiku
mctx
optax
```

JAX CPU build is sufficient — models are small (64 channels × 4 blocks) and MCTS depth is shallow (1–2 simulations at demo level).

---

## Bot Loading

On startup, `app.py` reads `bots.jsonl`, instantiates each bot lazily (first time the user selects it):

```python
# pseudo-code
def load_bot(entry: dict) -> BotPlayer:
    net_fn = make_network(entry["model_config"])
    params = load_checkpoint(entry["checkpoint_path"])  # orbax / pickle
    eval_cfg = entry["eval_config"]
    return BotPlayer(net_fn, params, eval_cfg)
```

`BotPlayer.pick_action(state)` runs Gumbel MCTS (`mctx.gumbel_muzero_policy`) with `num_simulations` from `eval_config` and returns the action with highest visit count.

Bot entries expose `name`, `icon`, and `description` for the UI.

---

## UI Layout

```
┌─────────────────────────────────────────────┐
│  🌰 Pungupua · Bao la Kiswahili             │
│                                             │
│  Opponent:  [🦆 bata_demo_1 ▾]  [New Game] │
│                                             │
│  ┌─── BOARD ───────────────────────────┐   │
│  │   8  7  6  5  4  3  2  1  ← Bot     │   │
│  │  [ b row — bot back               ] │   │
│  │  [ a row — bot front (nyumba=[5]) ] │   │
│  │  ─────────────────────────────────  │   │
│  │  [ A row — your front             ] │   │
│  │  │  ← 6 →  │  · 2 ·  │ [5]→  │ … │ │   │
│  │  [ B row — your back              ] │   │
│  │   1  2  3  4  5  6  7  8  ← You    │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  Stage: Namua  |  Stock — You: 22  Bot: 22  │
│                                             │
│  Selected: A5R      [ Play Move ]           │
│  (click ← or → on a highlighted cell)      │
│                                             │
│  Move log:                                  │
│  ┌─────────────────────────────────────┐   │
│  │  1. Bot: A3L   You: A5R            │   │
│  │  2. Bot: B2R   ...                 │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Board Component

Rendered as an HTML table inside a `gr.HTML` component. Each cell shows the seed count with directional arrow buttons flanking it for legal moves.

**Cell anatomy** (human rows only, legal cells):
```
┌───────────────┐
│  ←    6    →  │
└───────────────┘
  ▲          ▲
  │          └─ right-sow button (shown only if A6R is legal)
  └─ left-sow button (shown only if A6L is legal)
```

Non-legal cells and all bot cells render as plain seed counts — no arrows, no pointer cursor:
```
┌───────────────┐
│       2       │
└───────────────┘
```

The nyumba cell gets a coloured border regardless of legality.

**Nyumba-pending state**: when `nyumba_pending = True`, the normal arrow buttons are hidden and two large buttons appear below the board: **[ Stop ]** and **[ Continue ]**. Clicking either immediately submits that action (no confirm step needed — there are only two choices).

**Color scheme:**
- Bot side cells: muted teal (`#e0f2f1`)
- Human side cells (idle): muted amber (`#fff8e1`)
- Human side cells (legal, unselected): white with amber dashed border + pointer cursor
- Human side cells (selected): amber fill (`#ffb300`), white text
- Nyumba cell border: 2 px solid `#6d4c41` (dark brown)
- Bot side during bot's turn: cells gently pulse to signal "thinking"

### Move Selection Flow

1. Human clicks a `←` or `→` arrow on a legal cell.
2. The arrow's `onclick` calls `selectMove("A5R")` (a tiny inline JS function).
3. `selectMove` writes the move string to a **hidden** `gr.Textbox(elem_id="selected_move", visible=False)` and fires an `input` event on it.
4. The `selected_move` textbox `.input()` callback (Python) re-renders the board with the chosen cell highlighted and updates the **Selected** label above the Play button.
5. Human clicks **[ Play Move ]** to submit.

This two-step pattern (select then confirm) prevents fat-finger plays in a game where moves can't be undone.

```javascript
// Injected into the board HTML
function selectMove(move) {
  // update hidden Gradio textbox
  const tb = document.querySelector('#selected_move textarea');
  tb.value = move;
  tb.dispatchEvent(new Event('input', { bubbles: true }));
}
```

The hidden textbox's `.input()` event triggers a lightweight Python handler that only re-renders the board highlight — it does **not** advance the game state. Only the **Play Move** button does that.

### Game Status Bar

`gr.Markdown` strip between board and move area:
```
Stage: Namua  |  Stock — You: 18  Bot: 20  |  Your turn
```
During bot's turn:
```
Stage: Namua  |  Stock — You: 18  Bot: 20  |  Bot is thinking…
```
On terminal state:
```
Game over — You won! 🎉
```
or
```
Game over — Bot wins.
```

---

## Gradio State Machine

```
gr.State(GameState)      ← serialised as a Python dict (JAX arrays → numpy)
gr.State(BotPlayer)      ← loaded lazily, keyed by bot name
gr.State(str)            ← currently selected move string, e.g. "A5R" or ""
```

**Gradio components wired up:**
| Component | `elem_id` | Purpose |
|---|---|---|
| `gr.HTML` | `board` | Rendered board; re-emitted on every state change |
| `gr.Textbox(visible=False)` | `selected_move` | JS bridge: receives clicked move string |
| `gr.Markdown` | `status_bar` | Stage / stock / turn info |
| `gr.Markdown` | `selected_label` | "Selected: A5R" display above Play button |
| `gr.Button` | `play_btn` | Submits the selected move |
| `gr.Button` | `new_game_btn` | Resets game |
| `gr.Dropdown` | `bot_selector` | Choose opponent bot |

**Event handlers:**

1. **`bot_selector.change`** → re-renders bot description card only (no game reset).
2. **`new_game_btn.click`** → `reset_game(bot_name)` → fresh `GameState`, fresh board HTML (arrows shown), cleared selected move, updated status bar.
3. **`selected_move.input`** → `on_cell_click(move_str, state)` → re-renders board with chosen cell highlighted; updates `selected_label`; enables `play_btn`. Does **not** advance game state.
4. **`play_btn.click`** → `human_step(state, selected_move)`:
   - Validates move is still legal (guard against stale clicks).
   - Applies `game.step(state, action)`.
   - Renders board (no arrows, bot-thinking pulse).
   - Updates status bar → "Bot is thinking…"
   - Calls `bot_step(state)` synchronously.
   - Re-renders board with fresh arrows for next human turn.
   - Clears `selected_move`; disables `play_btn` until next click.
5. **Nyumba-pending**: board renders **Stop** / **Continue** buttons instead of arrows. These call `selectMove("stop")` / `selectMove("continue")` and immediately also trigger `play_btn.click` — no explicit confirm needed since there are only two choices and neither is ambiguous.

Bot step is synchronous (fast at 1–2 MCTS sims on CPU) — no streaming or async needed.

---

## Board HTML Renderer

`render_board(abs_board, legal_actions, selected_move, nyumba_active) -> str` returns an HTML string.

### Absolute board orientation

`GameState.board` is always stored from the **current player's** perspective and flips each turn. The renderer needs a stable orientation — human rows always at the bottom. We maintain:

```python
abs_board: np.ndarray  # shape (4, 8), always P0=human perspective
# abs_board[0] = human front row,  abs_board[1] = human back row
# abs_board[2] = bot front row,    abs_board[3] = bot back row
```

Recompute after every `game.step` call:

```python
if state.current_player == 0:   # human's turn: board is already from P0's view
    abs_board = np.array(state.board)
else:                            # bot's turn: board is from P1's view, flip back
    b = np.array(state.board)
    abs_board = np.stack([b[2, ::-1], b[3, ::-1], b[0, ::-1], b[1, ::-1]])
```

`legal_actions` is only passed to the renderer when it is the human's turn (current_player == 0). When it's the bot's turn, `legal_actions = []` so no arrows appear.

### Cell rendering logic

```python
def render_cell(seeds, row_label, col, legal_actions, selected_move, nyumba=False):
    left_move  = f"{row_label}{col+1}L"
    right_move = f"{row_label}{col+1}R"
    left_legal  = action_str_to_id(left_move)  in legal_actions
    right_legal = action_str_to_id(right_move) in legal_actions

    left_btn  = f'<button onclick="selectMove(\'{left_move}\')" '  \
                f'class="arrow{" selected" if left_move == selected_move else ""}">←</button>' \
                if left_legal else '<span class="arrow-spacer"></span>'
    right_btn = ... # symmetric

    seed_span = f'<span class="seeds">{seeds}</span>'
    cls = "cell legal" if (left_legal or right_legal) else "cell"
    if nyumba: cls += " nyumba"
    if left_move == selected_move or right_move == selected_move: cls += " selected"

    return f'<td class="{cls}">{left_btn}{seed_span}{right_btn}</td>'
```

Bot cells (rows `a`, `b`) are rendered with col indices reversed and no arrow buttons:

```python
def render_bot_cell(seeds, nyumba=False):
    cls = "cell bot" + (" nyumba" if nyumba else "")
    return f'<td class="{cls}"><span class="seeds">{seeds}</span></td>'
```

### CSS (inlined in the HTML string)

```css
.bao-board { border-collapse: collapse; margin: auto; }
.cell       { width: 70px; height: 60px; text-align: center;
              background: #fff8e1; border: 1px solid #ccc; }
.cell.bot   { background: #e0f2f1; }
.cell.legal { border: 2px dashed #ffb300; cursor: default; }
.cell.selected { background: #ffb300; }
.cell.nyumba   { border: 2px solid #6d4c41 !important; }
.seeds      { font-size: 1.3em; font-weight: bold; }
button.arrow { background: none; border: none; font-size: 1.1em;
               cursor: pointer; padding: 2px 4px; border-radius: 4px; }
button.arrow:hover   { background: #ffe082; }
button.arrow.selected { background: #fb8c00; color: white; }
.arrow-spacer { display: inline-block; width: 24px; }
```

---

## Checkpoint Storage

Options (in order of preference):

1. **Bundle in Space repo**: upload `checkpoints/` directly. Fine for small checkpoints (< 100 MB). HF Spaces has a 50 GB limit.
2. **HF Hub dataset**: store weights in a private or public dataset repo, `hf_hub_download` at load time. Better for large checkpoints or frequent updates.

For now, bundle directly — checkpoint files are small (ResNet-4 with 64 channels ≈ a few MB).

---

## Deployment Steps

1. Create a new HF Space: `Gradio` SDK, linked to this repo (or a fork).
2. Add `space_requirements.txt`.
3. Add the YAML front-matter to `README.md`.
4. Upload `checkpoints/` (if not already tracked by git lfs).
5. Write `app.py` following the design above.
6. Push — HF builds and serves automatically.

Optional: set `HF_HOME=/tmp` in Space secrets to avoid home-dir write issues on the container.

---

## Open Questions / Future Work

- **Think time slider**: expose `num_simulations` as a `gr.Slider` so users can make bots stronger/weaker on the fly (within a safe max, e.g. 32).
- **Mobile layout**: the 4×8 table is wide; consider a rotated 8×4 layout on narrow viewports via CSS `@media` query.
- **Move animation**: briefly flash the cell(s) touched by the bot's move before settling, to help the human see what changed.
- **Bot vs Bot mode**: add a "watch" button that plays out a full bot-vs-bot game with a per-step delay.
- **Multiple bots**: as more checkpoints are added to `bots.jsonl`, they appear automatically in the dropdown with zero code changes.
