import argparse
import json
import math
import random
from pathlib import Path

import plotly.graph_objects as go

from deck import draw_without_replacement, full_deck
from simulator import simulate_state, simulate_state_unknown_opponents

STREETS = [("preflop", 0), ("flop", 3), ("turn", 4), ("river", 5)]


def deal_random_hand(n_players, rng):
    cards_needed = (2 * n_players) + 5
    drawn = draw_without_replacement(full_deck(), cards_needed, rng)
    hole_cards = [drawn[2 * i : 2 * i + 2] for i in range(n_players)]
    board_full = drawn[2 * n_players :]
    labels = [f"Player {i + 1}" for i in range(n_players)]
    return hole_cards, board_full, labels


def deal_hero_and_board(rng):
    drawn = draw_without_replacement(full_deck(), 7, rng)
    hero_hole = drawn[:2]
    board_full = drawn[2:]
    return hero_hole, board_full


def safe_entropy(probs):
    total = sum(max(p, 0.0) for p in probs)
    if total <= 0:
        return 0.0
    normalized = [max(p, 0.0) / total for p in probs]
    return -sum(p * math.log2(p) for p in normalized if p > 0)


def stage_uncertainty_stats(result, labels):
    win_probs = result["win_prob"]
    ci95 = result["win_ci95"]
    entropy = safe_entropy(win_probs)
    entropy_max = math.log2(len(labels)) if len(labels) > 1 else 1.0
    entropy_ratio = entropy / entropy_max if entropy_max > 0 else 0.0
    ci_widths = [max(0.0, hi - lo) for lo, hi in ci95]
    avg_ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else 0.0
    return {
        "entropy_bits": entropy,
        "entropy_ratio": entropy_ratio,
        "avg_ci_width": avg_ci_width,
    }


def simulate_fixed_hand(rng, n_players, trials_by_stage):
    hole_cards, board_full, labels = deal_random_hand(n_players, rng)
    entropy = []
    entropy_norm = []
    avg_ci = []
    for stage_name, n_board in STREETS:
        board = board_full[:n_board]
        result = simulate_state(
            hole_cards=hole_cards,
            board=board,
            n_trials=trials_by_stage[stage_name],
            seed=rng.randint(0, 10**9),
        )
        stats = stage_uncertainty_stats(result, labels)
        entropy.append(stats["entropy_bits"])
        entropy_norm.append(stats["entropy_ratio"])
        avg_ci.append(stats["avg_ci_width"])
    return {"entropy": entropy, "entropy_norm": entropy_norm, "avg_ci": avg_ci}


def simulate_unknown_hand(rng, n_players, trials_by_stage):
    hero_hole, board_full = deal_hero_and_board(rng)
    labels = ["Hero"] + [f"Opponent {i}" for i in range(1, n_players)]
    entropy = []
    entropy_norm = []
    avg_ci = []
    for stage_name, n_board in STREETS:
        board = board_full[:n_board]
        result = simulate_state_unknown_opponents(
            hero_hole=hero_hole,
            n_players=n_players,
            board=board,
            n_trials=trials_by_stage[stage_name],
            seed=rng.randint(0, 10**9),
        )
        stats = stage_uncertainty_stats(result, labels)
        entropy.append(stats["entropy_bits"])
        entropy_norm.append(stats["entropy_ratio"])
        avg_ci.append(stats["avg_ci_width"])
    return {"entropy": entropy, "entropy_norm": entropy_norm, "avg_ci": avg_ci}


def mean_and_ci95(matrix):
    # matrix shape: [n_hands][4]
    n = len(matrix)
    if n == 0:
        return [0.0] * 4, [0.0] * 4
    means = []
    ci95 = []
    for i in range(4):
        vals = [row[i] for row in matrix]
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            se = math.sqrt(var / n)
            ci = 1.96 * se
        else:
            ci = 0.0
        means.append(mean)
        ci95.append(ci)
    return means, ci95


def count_upward_steps(trajectory):
    # Number of street transitions where entropy increases.
    return sum(1 for i in range(3) if trajectory[i + 1] > trajectory[i])


def monotone_nonincreasing(trajectory):
    return all(trajectory[i + 1] <= trajectory[i] for i in range(3))


def run_experiment(n_hands, n_players, seed, trials_by_stage):
    rng = random.Random(seed)

    fixed_norm = []
    fixed_bits = []
    fixed_ci = []

    unknown_norm = []
    unknown_bits = []
    unknown_ci = []

    for _ in range(n_hands):
        f = simulate_fixed_hand(rng, n_players, trials_by_stage)
        u = simulate_unknown_hand(rng, n_players, trials_by_stage)
        fixed_bits.append(f["entropy"])
        fixed_norm.append(f["entropy_norm"])
        fixed_ci.append(f["avg_ci"])
        unknown_bits.append(u["entropy"])
        unknown_norm.append(u["entropy_norm"])
        unknown_ci.append(u["avg_ci"])

    return {
        "fixed_bits": fixed_bits,
        "fixed_norm": fixed_norm,
        "fixed_ci": fixed_ci,
        "unknown_bits": unknown_bits,
        "unknown_norm": unknown_norm,
        "unknown_ci": unknown_ci,
    }


def plot_entropy_trend(output_dir, fixed_norm, unknown_norm):
    x = [s.title() for s, _ in STREETS]
    fixed_mean, fixed_ci95 = mean_and_ci95(fixed_norm)
    unknown_mean, unknown_ci95 = mean_and_ci95(unknown_norm)
    fixed_lo = [max(0.0, m - c) for m, c in zip(fixed_mean, fixed_ci95)]
    fixed_hi = [min(1.0, m + c) for m, c in zip(fixed_mean, fixed_ci95)]
    unknown_lo = [max(0.0, m - c) for m, c in zip(unknown_mean, unknown_ci95)]
    unknown_hi = [min(1.0, m + c) for m, c in zip(unknown_mean, unknown_ci95)]
    fixed_monotone_rate = sum(1 for row in fixed_norm if monotone_nonincreasing(row)) / len(fixed_norm)
    unknown_monotone_rate = sum(1 for row in unknown_norm if monotone_nonincreasing(row)) / len(
        unknown_norm
    )

    fig = go.Figure()
    # Confidence ribbons first so trend lines remain on top.
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=fixed_hi + fixed_lo[::-1],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.14)",
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
            name="Fixed 95% CI band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=unknown_hi + unknown_lo[::-1],
            fill="toself",
            fillcolor="rgba(234, 88, 12, 0.14)",
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
            name="Unknown 95% CI band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=fixed_mean,
            mode="lines+markers",
            name="Fixed dealt hands",
            line={"width": 4, "color": "#2563EB"},
            marker={"size": 10, "symbol": "circle", "line": {"width": 1, "color": "#1E3A8A"}},
            error_y={"type": "data", "array": fixed_ci95},
            text=[f"{v:.3f}" for v in fixed_mean],
            textposition="top center",
            hovertemplate="<b>%{x}</b><br>Fixed mean: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=unknown_mean,
            mode="lines+markers",
            name="Unknown opponents",
            line={"width": 4, "dash": "dash", "color": "#EA580C"},
            marker={"size": 10, "symbol": "diamond", "line": {"width": 1, "color": "#9A3412"}},
            error_y={"type": "data", "array": unknown_ci95},
            text=[f"{v:.3f}" for v in unknown_mean],
            textposition="bottom center",
            hovertemplate="<b>%{x}</b><br>Unknown mean: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            "Figure 1: Normalized Entropy Decreases Across Streets"
            "<br><sup>4 players, many random hands; shaded bands show mean \u00b1 95% CI across hands</sup>"
        ),
        yaxis_title="Normalized Entropy",
        xaxis_title="Street",
        template="plotly_white",
        height=560,
        margin={"l": 70, "r": 30, "t": 100, "b": 70},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "xanchor": "left", "x": 0.0},
        hovermode="x unified",
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#FFFFFF",
        annotations=[
            {
                "x": 0.02,
                "y": 0.03,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "align": "left",
                "bordercolor": "#CBD5E1",
                "borderwidth": 1,
                "borderpad": 8,
                "bgcolor": "rgba(255,255,255,0.92)",
                "font": {"size": 12, "color": "#0F172A"},
                "text": (
                    "Hypothesis check: uncertainty declines as board information increases."
                    "<br>"
                    f"Monotone non-increasing entropy rate: "
                    f"<b>Fixed {fixed_monotone_rate:.1%}</b>, "
                    f"<b>Unknown {unknown_monotone_rate:.1%}</b>."
                ),
            }
        ],
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#94A3B8")
    fig.update_yaxes(range=[0, 1.05], showline=True, linewidth=1, linecolor="#94A3B8", gridcolor="#E2E8F0")
    fig.write_html(str(output_dir / "fig1_entropy_trend.html"), include_plotlyjs="cdn")


def plot_ci_trend(output_dir, fixed_ci, unknown_ci):
    x = [s.title() for s, _ in STREETS]
    labels = [s.title() for s, _ in STREETS]
    fixed_mean, fixed_ci95 = mean_and_ci95(fixed_ci)
    unknown_mean, unknown_ci95 = mean_and_ci95(unknown_ci)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=fixed_mean,
            mode="lines+markers",
            name="Fixed dealt hands",
            line={"width": 3, "color": "#2563EB"},
            error_y={"type": "data", "array": fixed_ci95},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=unknown_mean,
            mode="lines+markers",
            name="Unknown opponents",
            line={"width": 3, "dash": "dash", "color": "#EA580C"},
            error_y={"type": "data", "array": unknown_ci95},
        )
    )
    fig.update_layout(
        title="Figure 2: Confidence-Interval Width by Street",
        yaxis_title="Average 95% CI Width",
        xaxis_title="Street",
        template="plotly_white",
        height=480,
    )
    fig.write_html(str(output_dir / "fig2_ci_width_trend.html"), include_plotlyjs="cdn")


def plot_monotonicity(output_dir, fixed_norm, unknown_norm):
    fixed_up = [count_upward_steps(row) for row in fixed_norm]
    unknown_up = [count_upward_steps(row) for row in unknown_norm]

    counts_fixed = [fixed_up.count(i) for i in range(4)]
    counts_unknown = [unknown_up.count(i) for i in range(4)]

    x = ["0", "1", "2", "3"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=counts_fixed, name="Fixed dealt hands"))
    fig.add_trace(go.Bar(x=x, y=counts_unknown, name="Unknown opponents"))
    fig.update_layout(
        barmode="group",
        title="Figure 3: Local Entropy Increases Are Occasional",
        xaxis_title="Number of upward entropy steps (out of 3 transitions)",
        yaxis_title="Number of hands",
        template="plotly_white",
        height=480,
    )
    fig.write_html(str(output_dir / "fig3_monotonicity_hist.html"), include_plotlyjs="cdn")


def plot_sample_trajectories(output_dir, fixed_norm, unknown_norm, max_lines=30):
    x = [s.title() for s, _ in STREETS]
    labels = [s.title() for s, _ in STREETS]
    fig = go.Figure()
    for row in fixed_norm[:max_lines]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=row,
                mode="lines",
                showlegend=False,
                line={"color": "rgba(37,99,235,0.2)", "width": 1.2},
            )
        )
    for row in unknown_norm[:max_lines]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=row,
                mode="lines",
                showlegend=False,
                line={"color": "rgba(234,88,12,0.2)", "width": 1.2, "dash": "dash"},
            )
        )
    fixed_mean, _ = mean_and_ci95(fixed_norm)
    unknown_mean, _ = mean_and_ci95(unknown_norm)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=fixed_mean,
            mode="lines+markers",
            name="Fixed mean",
            line={"color": "#1D4ED8", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=unknown_mean,
            mode="lines+markers",
            name="Unknown mean",
            line={"color": "#C2410C", "width": 3, "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Figure 4: Hand-to-Hand Variability with Mean Trend",
        yaxis_title="Normalized Entropy",
        xaxis_title="Street",
        template="plotly_white",
        height=500,
    )
    fig.update_yaxes(range=[0, 1.05])
    fig.write_html(str(output_dir / "fig4_entropy_trajectories.html"), include_plotlyjs="cdn")


def write_summary(output_dir, results, n_hands, n_players, trials_by_stage, seed):
    fixed_norm = results["fixed_norm"]
    unknown_norm = results["unknown_norm"]

    fixed_monotone = sum(1 for row in fixed_norm if monotone_nonincreasing(row)) / len(fixed_norm)
    unknown_monotone = sum(1 for row in unknown_norm if monotone_nonincreasing(row)) / len(unknown_norm)

    fixed_mean_norm, _ = mean_and_ci95(fixed_norm)
    unknown_mean_norm, _ = mean_and_ci95(unknown_norm)
    fixed_mean_ci, _ = mean_and_ci95(results["fixed_ci"])
    unknown_mean_ci, _ = mean_and_ci95(results["unknown_ci"])

    summary = {
        "n_hands": n_hands,
        "n_players": n_players,
        "seed": seed,
        "trials_by_stage": trials_by_stage,
        "monotone_nonincreasing_rate": {
            "fixed_dealt_hands": fixed_monotone,
            "unknown_opponents": unknown_monotone,
        },
        "mean_normalized_entropy_by_street": {
            "fixed_dealt_hands": dict(zip([s for s, _ in STREETS], fixed_mean_norm)),
            "unknown_opponents": dict(zip([s for s, _ in STREETS], unknown_mean_norm)),
        },
        "mean_avg_ci_width_by_street": {
            "fixed_dealt_hands": dict(zip([s for s, _ in STREETS], fixed_mean_ci)),
            "unknown_opponents": dict(zip([s for s, _ in STREETS], unknown_mean_ci)),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run many-hand entropy analysis for CS109 write-up.")
    parser.add_argument("--hands", type=int, default=120, help="Number of random hands to simulate per mode.")
    parser.add_argument("--players", type=int, default=4, help="Number of players (2-10).")
    parser.add_argument("--seed", type=int, default=109, help="Random seed for reproducibility.")
    parser.add_argument("--outdir", type=str, default="analysis_outputs", help="Output directory for figures.")
    parser.add_argument("--preflop-trials", type=int, default=1500)
    parser.add_argument("--flop-trials", type=int, default=1500)
    parser.add_argument("--turn-trials", type=int, default=1500)
    parser.add_argument("--river-trials", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_by_stage = {
        "preflop": args.preflop_trials,
        "flop": args.flop_trials,
        "turn": args.turn_trials,
        "river": args.river_trials,
    }

    results = run_experiment(
        n_hands=args.hands,
        n_players=args.players,
        seed=args.seed,
        trials_by_stage=trials_by_stage,
    )

    plot_entropy_trend(output_dir, results["fixed_norm"], results["unknown_norm"])
    plot_ci_trend(output_dir, results["fixed_ci"], results["unknown_ci"])
    plot_monotonicity(output_dir, results["fixed_norm"], results["unknown_norm"])
    plot_sample_trajectories(output_dir, results["fixed_norm"], results["unknown_norm"])
    write_summary(output_dir, results, args.hands, args.players, trials_by_stage, args.seed)

    print(f"Saved figures and summary to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
