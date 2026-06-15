"""Round-robin bot tournament for Bao la Kiswahili.

Each pair of bots plays as both player 0 and player 1 every round.
ELO is updated after every individual game. A markdown report is written
at the end with standings, head-to-head win rates, per-bot breakdowns,
and game-length statistics.

Usage (run from project root):
    python -m eval.tournament
    python -m eval.tournament --rounds 5 --games-per-pair 2
    python -m eval.tournament --bots bots.jsonl --output report.md --seed 42
"""

import argparse
import datetime
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bao import Game
from eval.bots import Bot, load_bot, parse_bots_file
from eval.report import md_table


# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update_elo(
    ratings: np.ndarray, idx_a: int, idx_b: int, score_a: float, k: float
) -> None:
    """score_a: 1=A wins, 0.5=draw, 0=A loses. Updates ratings in-place."""
    exp_a = _elo_expected(ratings[idx_a], ratings[idx_b])
    ratings[idx_a] += k * (score_a - exp_a)
    ratings[idx_b] += k * ((1.0 - score_a) - (1.0 - exp_a))


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------


def run_games_batch(
    bot_a: Bot,
    bot_b: Bot,
    game: Game,
    n_games: int,
    rng_key: jax.Array,
    max_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Play n_games with bot_a as player 0 and bot_b as player 1.

    Returns:
        winners: int32 array (n_games,) — 0=A won, 1=B won, -1=timeout draw
        lengths: int32 array (n_games,) — number of steps until termination
    """
    step_fn = jax.jit(jax.vmap(game.step))

    init = game.init()
    states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_games), init)

    done = np.zeros(n_games, dtype=bool)
    winners = np.full(n_games, -1, dtype=np.int32)
    lengths = np.full(n_games, max_steps, dtype=np.int32)

    for step_num in range(max_steps):
        if done.all():
            break

        rng_key, key_a, key_b = jax.random.split(rng_key, 3)

        actions_a = np.asarray(bot_a.get_actions(states, key_a))
        actions_b = np.asarray(bot_b.get_actions(states, key_b))

        # Select action based on whose turn it is
        is_a_turn = np.asarray(states.current_player) == 0
        actions = jnp.array(np.where(is_a_turn, actions_a, actions_b))

        states = step_fn(states, actions)

        w_np = np.asarray(states.winner)
        terminal = w_np >= 0
        newly_done = terminal & ~done

        lengths[newly_done] = step_num + 1
        winners[newly_done] = w_np[newly_done]
        done |= terminal

    return winners, lengths


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------


@dataclass
class MatchRecord:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    # Side-specific breakdowns (P0 = first mover, P1 = second mover)
    wins_as_p0: int = 0
    draws_as_p0: int = 0
    losses_as_p0: int = 0
    wins_as_p1: int = 0
    draws_as_p1: int = 0
    losses_as_p1: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def win_pct(self) -> float:
        return self.score / self.total if self.total else 0.0

    def _side_pct(self, w: int, d: int, losses: int) -> float:
        t = w + d + losses
        return (w + 0.5 * d) / t * 100 if t else 0.0

    @property
    def p0_pct(self) -> float:
        return self._side_pct(self.wins_as_p0, self.draws_as_p0, self.losses_as_p0)

    @property
    def p1_pct(self) -> float:
        return self._side_pct(self.wins_as_p1, self.draws_as_p1, self.losses_as_p1)


def run_tournament(
    bots: list[Bot],
    n_rounds: int,
    games_per_pair: int,
    rng_key: jax.Array,
    max_steps: int = 200,
    k_elo: float = 32.0,
) -> tuple[list[list[MatchRecord]], np.ndarray, list[int]]:
    """Run full round-robin tournament.

    Returns:
        record[i][j]: MatchRecord for bot i vs bot j
        elo: final ELO ratings (n_bots,)
        all_lengths: list of game lengths
    """
    game = Game()
    n = len(bots)
    record: list[list[MatchRecord]] = [
        [MatchRecord() for _ in range(n)] for _ in range(n)
    ]
    elo = np.full(n, 1500.0)
    all_lengths: list[int] = []

    total = n_rounds * n * (n - 1)
    idx = 0

    for round_num in range(n_rounds):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                idx += 1
                rng_key, subkey = jax.random.split(rng_key)
                t0 = time.time()
                winners, lengths = run_games_batch(
                    bots[i],
                    bots[j],
                    game,
                    games_per_pair,
                    subkey,
                    max_steps,
                )
                dt = time.time() - t0

                w_counts = {0: 0, 1: 0, -1: 0}
                for g in range(games_per_pair):
                    w = int(winners[g])
                    w_counts[w] += 1
                    all_lengths.append(int(lengths[g]))

                    if w == 0:  # bot_i (P0) wins
                        record[i][j].wins += 1
                        record[i][j].wins_as_p0 += 1
                        record[j][i].losses += 1
                        record[j][i].losses_as_p1 += 1
                        _update_elo(elo, i, j, 1.0, k_elo)
                    elif w == 1:  # bot_j (P1) wins
                        record[i][j].losses += 1
                        record[i][j].losses_as_p0 += 1
                        record[j][i].wins += 1
                        record[j][i].wins_as_p1 += 1
                        _update_elo(elo, i, j, 0.0, k_elo)
                    else:  # timeout draw
                        record[i][j].draws += 1
                        record[i][j].draws_as_p0 += 1
                        record[j][i].draws += 1
                        record[j][i].draws_as_p1 += 1
                        _update_elo(elo, i, j, 0.5, k_elo)

                print(
                    f"  [{idx:>{len(str(total))}}/{total}] "
                    f"Round {round_num + 1}  "
                    f"{bots[i].icon}{bots[i].name} (P0) vs "
                    f"{bots[j].icon}{bots[j].name} (P1)  "
                    f"→  {w_counts[0]}W / {w_counts[-1]}D / {w_counts[1]}L  "
                    f"({dt:.1f}s)"
                )

    return record, elo, all_lengths


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def generate_report(
    bots: list[Bot],
    record: list[list[MatchRecord]],
    elo: np.ndarray,
    all_lengths: list[int],
    n_rounds: int,
    games_per_pair: int,
) -> str:
    n = len(bots)
    rank_order = list(np.argsort(-elo))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total_games = sum(record[i][j].total for i in range(n) for j in range(n)) // 2

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Bao la Kiswahili — Tournament Report",
        "",
        f"**Date:** {now}  ",
        f"**Rounds:** {n_rounds}  |  "
        f"**Games per ordered pair / round:** {games_per_pair}  |  "
        f"**Total games:** {total_games}",
        "",
        "Each ordered pair (A as P0, B as P1) plays every round, "
        "so every unordered pair meets twice per round.",
        "",
    ]

    # ── Overall standings ─────────────────────────────────────────────────────
    lines += ["## Standings", ""]

    standing_rows = []
    for rank, idx in enumerate(rank_order, 1):
        tw = sum(record[idx][j].wins for j in range(n) if j != idx)
        td = sum(record[idx][j].draws for j in range(n) if j != idx)
        tl = sum(record[idx][j].losses for j in range(n) if j != idx)
        tg = tw + td + tl
        score_pct = (tw + 0.5 * td) / tg * 100 if tg else 0.0
        standing_rows.append(
            [
                str(rank),
                f"{bots[idx].icon} {bots[idx].name}",
                f"{elo[idx]:.0f}",
                str(tw),
                str(td),
                str(tl),
                str(tg),
                f"{score_pct:.1f}%",
            ]
        )

    lines += md_table(
        ["Rank", "Bot", "ELO", "W", "D", "L", "Games", "Score%"], standing_rows
    )
    lines.append("")

    # ── ELO change summary ────────────────────────────────────────────────────
    lines += ["## ELO Ratings", ""]

    elo_rows = []
    for idx in rank_order:
        delta = elo[idx] - 1500.0
        sign = "+" if delta >= 0 else ""
        elo_rows.append(
            [
                f"{bots[idx].icon} {bots[idx].name}",
                f"{elo[idx]:.1f}",
                f"{sign}{delta:.1f}",
            ]
        )

    lines += md_table(["Bot", "ELO", "Δ from 1500"], elo_rows)
    lines.append("")

    # ── Head-to-head matrix ───────────────────────────────────────────────────
    lines += [
        "## Head-to-Head Win Rates",
        "",
        "*Row vs column: score% for the **row bot** across all games against the column bot.*",
        "",
    ]

    h2h_headers = ["Bot \\ Opponent"] + [
        f"{bots[j].icon} {bots[j].name}" for j in range(n)
    ]
    h2h_rows = []
    for i in range(n):
        row = [f"{bots[i].icon} {bots[i].name}"]
        for j in range(n):
            if i == j:
                row.append("—")
            else:
                r = record[i][j]
                pct = r.win_pct * 100
                row.append(f"{pct:.1f}% ({r.wins}W {r.draws}D {r.losses}L)")
        h2h_rows.append(row)

    lines += md_table(h2h_headers, h2h_rows)
    lines.append("")

    # ── Per-bot detail ────────────────────────────────────────────────────────
    lines += ["## Per-Bot Details", ""]

    for idx in rank_order:
        bot = bots[idx]
        tw = sum(record[idx][j].wins for j in range(n) if j != idx)
        td = sum(record[idx][j].draws for j in range(n) if j != idx)
        tl = sum(record[idx][j].losses for j in range(n) if j != idx)
        tg = tw + td + tl
        overall_pct = (tw + 0.5 * td) / tg * 100 if tg else 0.0

        p0_w = sum(record[idx][j].wins_as_p0 for j in range(n) if j != idx)
        p0_d = sum(record[idx][j].draws_as_p0 for j in range(n) if j != idx)
        p0_l = sum(record[idx][j].losses_as_p0 for j in range(n) if j != idx)
        p0_g = p0_w + p0_d + p0_l
        p0_pct = (p0_w + 0.5 * p0_d) / p0_g * 100 if p0_g else 0.0

        p1_w = sum(record[idx][j].wins_as_p1 for j in range(n) if j != idx)
        p1_d = sum(record[idx][j].draws_as_p1 for j in range(n) if j != idx)
        p1_l = sum(record[idx][j].losses_as_p1 for j in range(n) if j != idx)
        p1_g = p1_w + p1_d + p1_l
        p1_pct = (p1_w + 0.5 * p1_d) / p1_g * 100 if p1_g else 0.0

        lines += [
            f"### {bot.icon} {bot.name}",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Description | {bot.description} |",
            f"| Checkpoint | `{bot.checkpoint_path}` |",
            f"| Simulations | {bot.num_simulations} |",
            f"| Gumbel scale | {bot.gumbel_scale} |",
            f"| Final ELO | **{elo[idx]:.0f}** |",
            f"| Overall record | {tw}W / {td}D / {tl}L ({overall_pct:.1f}%) |",
            f"| As P0 (first mover) | {p0_w}W / {p0_d}D / {p0_l}L ({p0_pct:.1f}%) |",
            f"| As P1 (second mover) | {p1_w}W / {p1_d}D / {p1_l}L ({p1_pct:.1f}%) |",
            "",
        ]

        opp_rows = []
        for j in rank_order:
            if j == idx:
                continue
            r = record[idx][j]
            opp_rows.append(
                [
                    f"{bots[j].icon} {bots[j].name}",
                    str(r.wins),
                    str(r.draws),
                    str(r.losses),
                    str(r.total),
                    f"{r.win_pct * 100:.1f}%",
                    f"{r.p0_pct:.1f}%",
                    f"{r.p1_pct:.1f}%",
                ]
            )

        lines += md_table(
            ["Opponent", "W", "D", "L", "Games", "Score%", "as P0", "as P1"], opp_rows
        )
        lines.append("")

    # ── Game length statistics ────────────────────────────────────────────────
    if all_lengths:
        arr = np.array(all_lengths, dtype=float)
        lines += ["## Game Length Statistics", ""]
        lines += md_table(
            ["Statistic", "Value"],
            [
                ["Mean", f"{arr.mean():.1f}"],
                ["Median", f"{float(np.median(arr)):.1f}"],
                ["Std dev", f"{arr.std():.1f}"],
                ["Min", f"{int(arr.min())}"],
                ["Max", f"{int(arr.max())}"],
                ["Timeouts (= max steps)", str(int((arr == max(all_lengths)).sum()))],
            ],
        )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a round-robin tournament between Bao bots."
    )
    parser.add_argument(
        "--bots",
        default="bots.jsonl",
        help="Path to bots JSONL file (default: bots.jsonl)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of tournament rounds (default: 10)",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=1,
        help="Games per ordered pair per round (default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max moves per game before declaring a draw (default: 200)",
    )
    parser.add_argument(
        "--elo-k", type=float, default=32.0, help="ELO K-factor (default: 32)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--output",
        default="tournament_report.md",
        help="Output markdown file (default: tournament_report.md)",
    )
    args = parser.parse_args()

    print(f"Loading bots from {args.bots} …")
    bot_dicts = parse_bots_file(args.bots)

    if len(bot_dicts) < 2:
        print("Need at least 2 bots for a tournament.", file=sys.stderr)
        sys.exit(1)

    game = Game()
    bots: list[Bot] = []
    for d in bot_dicts:
        print(f"  Loading {d.get('icon', '')} {d['name']} … ", end="", flush=True)
        bot = load_bot(d, game)
        bots.append(bot)
        print("ok")

    n = len(bots)
    total_games = args.rounds * n * (n - 1) * args.games_per_pair
    print(
        f"\nTournament: {n} bots  |  {args.rounds} rounds  |  "
        f"{args.games_per_pair} game(s)/ordered-pair/round  |  "
        f"{total_games} total games\n"
    )

    rng_key = jax.random.PRNGKey(args.seed)
    t_start = time.time()

    record, elo, all_lengths = run_tournament(
        bots=bots,
        n_rounds=args.rounds,
        games_per_pair=args.games_per_pair,
        rng_key=rng_key,
        max_steps=args.max_steps,
        k_elo=args.elo_k,
    )

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s  ({elapsed / total_games:.2f}s/game)")

    print("\nFinal standings:")
    for rank, idx in enumerate(np.argsort(-elo), 1):
        tw = sum(record[idx][j].wins for j in range(n) if j != idx)
        td = sum(record[idx][j].draws for j in range(n) if j != idx)
        tl = sum(record[idx][j].losses for j in range(n) if j != idx)
        print(
            f"  #{rank}  {bots[idx].icon} {bots[idx].name}  ELO {elo[idx]:.0f}  "
            f"({tw}W / {td}D / {tl}L)"
        )

    report = generate_report(
        bots,
        record,
        elo,
        all_lengths,
        args.rounds,
        args.games_per_pair,
    )
    Path(args.output).write_text(report)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
