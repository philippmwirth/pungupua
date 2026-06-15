"""Opening analysis for Bao la Kiswahili.

Simulates the first N moves from the standard opening position using deep
AlphaZero MCTS (gumbel_scale=0, deterministic), and produces a move-by-move
report with Q-value rankings and a structured placeholder for move commentary.

Usage:
    python -m eval.opening_analysis --bot bata_demo_greedy --moves 10
    python -m eval.opening_analysis --bot bata_demo_greedy --moves 10 \\
        --simulations 800 --output opening_analysis.md
"""

import argparse
import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import mctx
import numpy as np

from bao import Game, NUM_ACTIONS
from eval.board import action_name, board_str
from eval.bots import Bot, load_bot, parse_bots_file
from eval.report import md_table


# ---------------------------------------------------------------------------
# Q-value extraction
# ---------------------------------------------------------------------------


def _top5(policy_output, mask) -> list[dict]:
    """Return up to 5 legal, visited actions ranked by policy weight (desc).

    Ranking by policy weight (the MCTS improved policy) keeps rank 1 consistent
    with the picked move, which is always argmax(action_weights). Q-values are
    included as supplementary information.
    Q(s,a) = r(s,a) + γ·V(s') from the root node of the search tree.
    """
    ri = mctx.Tree.ROOT_INDEX
    tree = policy_output.search_tree
    q_vals = np.array(
        tree.children_rewards[0, ri]
        + tree.children_discounts[0, ri] * tree.children_values[0, ri]
    )
    visits = np.array(tree.children_visits[0, ri], dtype=int)
    weights = np.array(policy_output.action_weights[0])
    legal = np.array(mask)

    candidates = [a for a in range(NUM_ACTIONS) if legal[a] and visits[a] > 0]
    candidates.sort(key=lambda a: (weights[a], q_vals[a]), reverse=True)
    return [
        {
            "action": a,
            "q_value": float(q_vals[a]),
            "policy": float(weights[a]),
            "visits": int(visits[a]),
        }
        for a in candidates[:5]
    ]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    bot: Bot,
    num_moves: int,
    num_simulations: int,
) -> str:
    game = Game()
    state = game.init()
    player_labels = ["P0", "P1"]

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        "# Bao la Kiswahili — Opening Analysis",
        "",
        f"**Bot:** {bot.icon} {bot.name}  ",
        f"**Simulations per move:** {num_simulations}  ",
        f"**Moves analysed:** {num_moves}  ",
        f"**Date:** {now}",
        "",
        "---",
        "",
    ]

    for move_num in range(1, num_moves + 1):
        if bool(game.is_terminal(state)):
            lines += [f"_Game ended after move {move_num - 1}._", ""]
            break

        player = player_labels[int(state.current_player)]
        print(f"  Move {move_num}/{num_moves} ({player})...", flush=True)

        # Run MCTS with override num_simulations
        batched = jax.tree_util.tree_map(lambda x: x[None], state)
        from eval.bots import run_mcts
        policy_output = run_mcts(
            bot.model,
            bot.forward,
            game,
            batched,
            jax.random.PRNGKey(0),
            num_simulations=num_simulations,
            gumbel_scale=0.0,
        )

        mask = game.legal_action_mask(state)
        top5 = _top5(policy_output, mask)
        action = int(policy_output.action[0])
        # Use the best Q over visited legal actions rather than the raw
        # node_values average.  node_values[root] is a backprop running average
        # over ALL simulations; for 2-action nyumba decisions Gumbel MuZero
        # allocates equal visits to both options, so the average is dragged down
        # by the worst action and is misleading as a position evaluation.
        ri = mctx.Tree.ROOT_INDEX
        _q = (
            np.array(policy_output.search_tree.children_rewards[0, ri])
            + np.array(policy_output.search_tree.children_discounts[0, ri])
            * np.array(policy_output.search_tree.children_values[0, ri])
        )
        _vis = np.array(policy_output.search_tree.children_visits[0, ri])
        _legal = np.array(mask)
        root_value = float(
            max(_q[a] for a in range(NUM_ACTIONS) if _legal[a] and _vis[a] > 0)
        )

        # --- Board position ---
        lines += [
            f"## Move {move_num} — {player} to play",
            "",
            "```",
            board_str(state),
            "```",
            "",
        ]

        # --- Top-5 table (ranked by policy weight; rank 1 = picked move) ---
        lines += ["**Top 5 moves by policy weight:**", ""]
        table_rows = [
            [
                str(rank),
                action_name(r["action"]),
                f"{r['q_value']:+.4f}",
                f"{r['policy']:.4f}",
                str(r["visits"]),
            ]
            for rank, r in enumerate(top5, 1)
        ]
        lines += md_table(["Rank", "Move", "Q-value", "Policy", "Visits"], table_rows)
        lines += [""]

        # --- Picked move ---
        lines += [
            f"**Root value:** {root_value:+.4f}  ",
            f"**Picked move: {action_name(action)}**",
            "",
        ]

        # --- Move analysis placeholder ---
        lines += [
            "### Move analysis",
            "",
            f"> **TODO: Claude, analyse move {action_name(action)}"
            f" (move {move_num}, {player} to play).**",
            ">",
            "> Please cover the following points:",
            ">",
            "> 1. **Capture or takasa?** Does this move capture opponent seeds?",
            ">    If so, which opponent front-row hole is targeted and how many",
            ">    seeds are taken? If the source is a back-row hole, describe the",
            ">    do-continue chain that produces the capture.",
            ">",
            "> 2. **Q-value context.** How does the picked move's Q-value compare",
            ">    to the alternatives in the table? Is this a clear best move or",
            ">    a close call? If the gap to rank 2 is small, explain what the",
            ">    alternative offers and why the bot prefers this move.",
            ">",
            "> 3. **Positional consequence.** Describe the board structure after",
            ">    this move: which holes are strengthened or weakened, what new",
            ">    capture threats arise, and whether the opponent's front row is",
            ">    more or less exposed.",
            ">",
            "> 4. **Nyumba relevance.** Does this move interact with either",
            ">    player's nyumba — sowing into it, threatening it, or emptying",
            ">    it? If so, what are the strategic implications?",
            ">",
            f"> 5. **Opening assessment.** Given this is move {move_num} of the",
            ">    opening, is this a standard developing move, an early attack,",
            ">    or an unusual choice? Would a Bao expert approve?",
            "",
            "---",
            "",
        ]

        state = game.step(state, jnp.int32(action))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Bao opening analysis report."
    )
    parser.add_argument(
        "--bot",
        required=True,
        help="Bot name from bots.jsonl (e.g. bata_demo_greedy)",
    )
    parser.add_argument(
        "--moves",
        type=int,
        default=10,
        help="Number of moves to analyse (default: 10)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="MCTS simulations per move (default: 800)",
    )
    parser.add_argument(
        "--bots",
        default="bots.jsonl",
        help="Path to bots JSONL file (default: bots.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="opening_analysis.md",
        help="Output markdown file (default: opening_analysis.md)",
    )
    args = parser.parse_args()

    bot_dicts = parse_bots_file(args.bots)
    bot_dict = next((d for d in bot_dicts if d["name"] == args.bot), None)
    if bot_dict is None:
        available = ", ".join(d["name"] for d in bot_dicts)
        raise ValueError(f"Bot '{args.bot}' not found. Available: {available}")

    print(f"Loading bot '{args.bot}'...")
    bot = load_bot(bot_dict)
    print(f"Loaded {bot.icon} {bot.name}")
    print(f"Simulations per move: {args.simulations}")

    report = generate_report(bot, args.moves, args.simulations)
    Path(args.output).write_text(report)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
