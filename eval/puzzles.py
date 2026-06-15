"""Puzzle report for Bao la Kiswahili.

For each puzzle, renders the board and shows each bot's greedy best move.

Usage:
    python -m eval.puzzles
    python -m eval.puzzles --bots bots.jsonl --output puzzles_report.md
"""

import argparse
import datetime
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx

from bao import Game
from bao.state import GameState
from eval.board import action_name, board_str
from eval.bots import Bot, load_bot, parse_bots_file
from eval.report import md_table


# ---------------------------------------------------------------------------
# Puzzle definition
# ---------------------------------------------------------------------------


class Puzzle(NamedTuple):
    name: str
    description: str
    state: GameState
    correct_actions: frozenset  # set of action indices that are correct


def _state(board, *, stock=(0, 0), stage=1, na=(False, False)):
    return GameState(
        board=jnp.array(board, jnp.int16),
        stock=jnp.array(stock, jnp.int16),
        current_player=jnp.int32(0),
        stage=jnp.int32(stage),
        winner=jnp.int32(-1),
        nyumba_active=jnp.array(na),
        nyumba_pending=jnp.bool_(False),
        pending_direction=jnp.int32(1),
    )


# ---------------------------------------------------------------------------
# Puzzle list — add your puzzles here
# ---------------------------------------------------------------------------

PUZZLES: list[Puzzle] = [
    Puzzle(
        name="test_1",
        description="",
        state=_state(
            [
                [2, 0, 1, 2, 0, 1, 1, 1],  # cur front
                [1, 2, 4, 5, 3, 3, 3, 2],
                [3, 0, 3, 0, 0, 5, 1, 0],  # opp front
                [2, 2, 4, 6, 0, 0, 3, 4],
            ]
        ),
        correct_actions=frozenset(),
    ),
    Puzzle(
        name="test_2",
        description="",
        state=_state(
            [
                [0, 2, 0, 12, 0, 2, 2, 1],  # cur front
                [0, 0, 3, 3, 2, 2, 1, 1],
                [0, 3, 0, 2, 4, 2, 3, 2],  # opp front
                [0, 0, 1, 2, 5, 1, 0, 8],
            ]
        ),
        correct_actions=frozenset(),
    ),
    Puzzle(
        name="test_3",
        description="",
        state=_state(
            [
                [1, 6, 3, 1, 0, 5, 4, 0],  # cur front
                [2, 1, 0, 5, 0, 3, 0, 1],
                [0, 0, 0, 0, 9, 0, 0, 1],  # opp front
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        correct_actions=frozenset(),
    ),
]


def _bot_search(bot: Bot, game: Game, state: GameState):
    """Run deterministic MCTS for a bot on a single state; return (action, prob, value)."""
    policy_output = bot.search(game, state, jax.random.PRNGKey(0))
    action = int(policy_output.action[0])
    prob = float(policy_output.action_weights[0, action])
    value = float(policy_output.search_tree.node_values[0, mctx.Tree.ROOT_INDEX])
    return action, prob, value


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(puzzles: list[Puzzle], bots_file: str) -> str:
    game = Game()
    bot_dicts = parse_bots_file(bots_file)

    print(f"Loading {len(bot_dicts)} bot(s)...")
    bots: list[Bot] = []
    for d in bot_dicts:
        bots.append(load_bot(d))
        print(f"  Loaded {d.get('icon', '')} {d['name']})")

    lines: list[str] = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    bot_names = ", ".join(f"{b.icon} {b.name}" for b in bots)
    lines += [
        "# Bao la Kiswahili — Puzzle Report",
        "",
        f"**Date:** {now}  ",
        f"**Bots:** {bot_names}  ",
        f"**Puzzles:** {len(puzzles)}",
        "",
        "---",
        "",
    ]

    for i, puzzle in enumerate(puzzles, 1):
        state = puzzle.state
        mask = game.legal_action_mask(state)
        legal = sorted(int(a) for a in jnp.where(mask)[0].tolist())

        lines += [f"## {i}. {puzzle.name}", ""]
        if puzzle.description:
            lines += [puzzle.description, ""]

        lines += ["```", board_str(state), "```", ""]

        legal_str = ", ".join(f"{action_name(a)} ({a})" for a in legal)
        lines += [f"**Legal actions:** {legal_str}", ""]

        if puzzle.correct_actions:
            correct_str = ", ".join(
                f"{action_name(a)} ({a})" for a in sorted(puzzle.correct_actions)
            )
            lines += [f"**Correct:** {correct_str}", ""]

        table_rows = []
        for bot in bots:
            action, prob, value = _bot_search(bot, game, state)
            marker = (
                " ✓"
                if puzzle.correct_actions and action in puzzle.correct_actions
                else ""
            )
            next_state = game.step(state, jnp.int32(action))
            result = "Win" if bool(next_state.winner == state.current_player) else "—"
            table_rows.append(
                [
                    f"{bot.icon} {bot.name}",
                    f"{action_name(action)} ({action}){marker}",
                    f"{prob:.3f}",
                    f"{value:+.3f}",
                    result,
                ]
            )

        lines += md_table(["Bot", "Best move", "Prob", "Value", "Result"], table_rows)
        lines += ["", "---", ""]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Bao puzzle report.")
    parser.add_argument(
        "--bots",
        default="bots.jsonl",
        help="Path to bots JSONL file (default: bots.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="puzzles_report.md",
        help="Output markdown file (default: puzzles_report.md)",
    )
    args = parser.parse_args()

    report = generate_report(PUZZLES, args.bots)
    Path(args.output).write_text(report)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
