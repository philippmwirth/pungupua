"""Board rendering utilities for Bao la Kiswahili."""

import numpy as np

from bao.state import GameState


def action_name(a: int) -> str:
    """Return action in play.py notation: A5R, B3L, stop, continue."""
    if a == 32:
        return "stop"
    if a == 33:
        return "continue"
    letter = "A" if a < 16 else "B"
    rem = a if a < 16 else a - 16
    col = rem // 2
    direction = "R" if rem % 2 == 1 else "L"
    return f"{letter}{col + 1}{direction}"


def _row_str(row, nyumba_col: int | None = None) -> str:
    cells = []
    for i in range(8):
        v = int(row[i])
        if nyumba_col is not None and i == nyumba_col:
            cells.append(f"[{v:2d}]")
        else:
            cells.append(f" {v:2d} ")
    return "  ".join(cells)


def board_str(state: GameState) -> str:
    b = np.array(state.board)
    na = np.array(state.nyumba_active)
    stock = np.array(state.stock)
    stage = "Namua" if int(state.stage) == 0 else "Mtaji"

    opp_nyumba_mark = 3 if na[1] else None  # board[2, 3] = opponent nyumba
    cur_nyumba_mark = 4 if na[0] else None  # board[0, 4] = current player nyumba

    cur_header = "  ".join(f"  {c + 1} " for c in range(8))
    opp_header = "  ".join(f"  {8 - c} " for c in range(8))

    lines = [
        f"Stage: {stage}   Stock — cur: {int(stock[0])}  opp: {int(stock[1])}",
        f"          {opp_header}           ← opponent",
        "",
        f"  b (back)  {_row_str(b[3])}",
        f"  a (front) {_row_str(b[2], opp_nyumba_mark)}",
        "",
        f"  A (front) {_row_str(b[0], cur_nyumba_mark)}",
        f"  B (back)  {_row_str(b[1])}",
        f"          {cur_header}           ← current player",
    ]
    return "\n".join(lines)
