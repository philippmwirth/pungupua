"""Terminal play-test for Bao la Kiswahili.

Usage:
    python play.py

Player 0 is always "you" (human), player 1 is the computer.
Moves are entered in notation: A5R, B3L, stop, continue.
"""
import random
import sys

import jax.numpy as jnp
import numpy as np

from src.bao.game import Game
from src.bao.state import GameState, NYUMBA_STOP, NYUMBA_CONTINUE, NYUMBA_COL
from src.bao.board import action_to_row_col_dir


PLAYER_NAMES = {0: "You", 1: "Computer"}
STAGE_NAMES  = {0: "Namua", 1: "Mtaji"}


# ── Display ──────────────────────────────────────────────────────────────────

def _row_str(row: np.ndarray, nyumba_col: int | None = None) -> str:
    """Render a row of 8 cells. Marks the nyumba with brackets."""
    cells = []
    for i in range(8):
        v = int(row[i])
        if nyumba_col is not None and i == nyumba_col:
            cells.append(f"[{v:2d}]")
        else:
            cells.append(f" {v:2d} ")
    return "  ".join(cells)


def _hole_label(row_idx: int, col: int) -> str:
    """Human-readable hole label: A1..A8 for front row, B1..B8 for back row."""
    letter = "A" if row_idx == 0 else "B"
    return f"{letter}{col + 1}"


def display_board(state: GameState, player_id: int) -> None:
    board = np.array(state.board)
    na    = np.array(state.nyumba_active)
    stock = np.array(state.stock)
    cp    = int(state.current_player)
    stage = int(state.stage)

    # Rows from current player's (board-owner's) perspective:
    #   board[0] = current player's front row
    #   board[1] = current player's back row
    #   board[2] = opponent's front row  (col i opposes board[0,i])
    #   board[3] = opponent's back row

    # Determine who owns rows 0/1 and 2/3
    cur_name  = PLAYER_NAMES[cp]
    opp_name  = PLAYER_NAMES[1 - cp]

    # board[2, col] is stored such that col i directly opposes board[0, i].
    # board[2, 3] is always the opponent's nyumba (hole 5 from their left).
    # We display opponent rows as-is (no reversal); their hole numbers run
    # right-to-left from our perspective, so we label their header 8→1.

    opp_nyumba_mark = 3 if na[1] else None   # board[2, 3] = opponent nyumba
    cur_nyumba_mark = 4 if na[0] else None   # board[0, 4] = current player nyumba

    cur_header = "  ".join(f"  {c+1} " for c in range(8))
    opp_header = "  ".join(f"  {8-c} " for c in range(8))

    print()
    print("=" * 76)
    print(f"  Stage: {STAGE_NAMES[stage]}   |   "
          f"Current turn: {cur_name} (P{cp})   |   "
          f"Stock — {cur_name}: {stock[0]}  {opp_name}: {stock[1]}")
    if state.nyumba_pending:
        print(f"  ** NYUMBA PENDING: choose 'stop' (32) or 'continue' (33) **")
    print("=" * 76)
    print(f"          {opp_header}           ← {opp_name} (P{1-cp})")
    print()
    print(f"  b (back)  {_row_str(board[3])}")
    print(f"  a (front) {_row_str(board[2], opp_nyumba_mark)}")
    print()
    print(f"  A (front) {_row_str(board[0], cur_nyumba_mark)}")
    print(f"  B (back)  {_row_str(board[1])}")
    print(f"          {cur_header}           ← {cur_name} (P{cp})")
    print("=" * 76)


# ── Action codec ─────────────────────────────────────────────────────────────

def action_to_str(action: int) -> str:
    if action == NYUMBA_STOP:
        return "stop"
    if action == NYUMBA_CONTINUE:
        return "continue"
    row = 0 if action < 16 else 1
    rem = action if action < 16 else action - 16
    col = rem // 2
    direction = rem % 2
    letter = "A" if row == 0 else "B"
    dir_char = "R" if direction == 1 else "L"
    return f"{letter}{col + 1}{dir_char}"


def str_to_action(s: str) -> int | None:
    s = s.strip().upper()
    if s in ("STOP", "32"):
        return NYUMBA_STOP
    if s in ("CONTINUE", "CONT", "33"):
        return NYUMBA_CONTINUE
    if len(s) < 3:
        return None
    row_char = s[0]
    dir_char = s[-1]
    hole_str = s[1:-1]
    if row_char not in ("A", "B") or dir_char not in ("L", "R"):
        return None
    try:
        hole = int(hole_str)
    except ValueError:
        return None
    if not (1 <= hole <= 8):
        return None
    col = hole - 1
    direction = 1 if dir_char == "R" else 0
    offset = 0 if row_char == "A" else 16
    return offset + col * 2 + direction


# ── Legal moves ───────────────────────────────────────────────────────────────

def list_legal_moves(game: Game, state: GameState) -> list[int]:
    mask = np.array(game.legal_action_mask(state))
    return [i for i, legal in enumerate(mask) if legal]


def display_legal_moves(legal: list[int]) -> None:
    labels = [action_to_str(a) for a in legal]
    print(f"  Legal moves: {', '.join(labels)}")


# ── Human turn ────────────────────────────────────────────────────────────────

def human_turn(game: Game, state: GameState) -> GameState:
    legal = list_legal_moves(game, state)
    display_legal_moves(legal)
    while True:
        raw = input("  Your move: ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("Goodbye.")
            sys.exit(0)
        action = str_to_action(raw)
        if action is None:
            print("  Invalid format. Use e.g. A5R, B3L, stop, continue.")
            continue
        if action not in legal:
            print(f"  '{raw}' is not a legal move. Try one of: {', '.join(action_to_str(a) for a in legal)}")
            continue
        state = game.step(state, jnp.int32(action))
        print(f"\n  You played: {action_to_str(action)}")
        display_board(state, int(state.current_player))
        return state


# ── Computer turn ─────────────────────────────────────────────────────────────

def computer_turn(game: Game, state: GameState) -> GameState:
    legal = list_legal_moves(game, state)
    action = random.choice(legal)
    label  = action_to_str(action)
    print(f"\n  Computer's move: {label}")
    input("  Press Enter to confirm... ")
    state = game.step(state, jnp.int32(action))
    display_board(state, int(state.current_player))
    return state


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    game  = Game()
    state = game.init()

    print("\nBao la Kiswahili — terminal play-test")
    print("You are Player 0. Computer is Player 1.")
    print("Enter moves as A5R (front row, hole 5, right), B3L (back row, hole 3, left),")
    print("'stop' or 'continue' for nyumba decisions. 'q' to quit.")

    display_board(state, int(state.current_player))

    while not game.is_terminal(state):
        cp = int(state.current_player)
        if cp == 0:
            state = human_turn(game, state)
        else:
            state = computer_turn(game, state)

        if game.is_terminal(state):
            break

    winner = int(state.winner)
    print(f"\n{'=' * 76}")
    print(f"  Game over! Winner: {PLAYER_NAMES[winner]} (P{winner})")
    print(f"{'=' * 76}\n")


if __name__ == "__main__":
    main()
