import math
import matplotlib.pyplot as plt


def _style():
    return {
        "win": "#6366F1",
        "tie": "#14B8A6",
        "entropy": "#8B5CF6",
        "entropy_norm": "#F59E0B",
        "grid": "#E2E8F0",
        "text_muted": "#475569",
        "bg": "#F1F5F9",
        "panel": "#FFFFFF",
        "title": "#0F172A",
        "ci": "#2563EB",
    }


def _player_colors(n_players):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n_players)]


def _apply_global_plot_style():
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "figure.titlesize": 16,
            "figure.facecolor": _style()["bg"],
            "axes.facecolor": _style()["panel"],
            "savefig.facecolor": _style()["bg"],
            "axes.edgecolor": "#94A3B8",
            "grid.color": _style()["grid"],
            "grid.alpha": 0.45,
        }
    )


def _beautify_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)


def _try_fullscreen(fig):
    manager = getattr(fig.canvas, "manager", None)
    if manager is None:
        return
    # Try common backend-specific maximize/fullscreen paths.
    try:
        if hasattr(manager, "window"):
            window = manager.window
            if hasattr(window, "showMaximized"):
                window.showMaximized()
                return
            if hasattr(window, "state"):
                window.state("zoomed")
                return
        if hasattr(manager, "full_screen_toggle"):
            manager.full_screen_toggle()
    except Exception:
        # Keep plotting resilient if backend does not support fullscreen.
        return


def _safe_entropy(probs):
    total = sum(max(p, 0.0) for p in probs)
    if total <= 0:
        return 0.0
    normalized = [max(p, 0.0) / total for p in probs]
    # Shannon entropy (bits): measures how spread-out win probability is.
    return -sum(p * math.log2(p) for p in normalized if p > 0)


def _entropy_series(stage_records):
    entropies = []
    for record in stage_records:
        entropies.append(_safe_entropy(record["win_prob"]))
    return entropies


def stage_uncertainty_stats(result, labels):
    win_probs = result["win_prob"]
    ci95 = result["win_ci95"]
    # Entropy at this street, using players' win-probability distribution.
    entropy = _safe_entropy(win_probs)
    # Maximum possible entropy for this table size (uniform distribution).
    entropy_max = math.log2(len(labels)) if len(labels) > 1 else 1.0
    entropy_ratio = entropy / entropy_max if entropy_max > 0 else 0.0
    # 95% CI width per player (hi - lo) is our uncertainty magnitude.
    ci_widths = [max(0.0, hi - lo) for lo, hi in ci95]
    avg_ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else 0.0
    max_ci_width = max(ci_widths) if ci_widths else 0.0
    return {
        "entropy_bits": entropy,
        "entropy_max_bits": entropy_max,
        "entropy_ratio": entropy_ratio,
        "avg_ci_width": avg_ci_width,
        "max_ci_width": max_ci_width,
        "ci_widths": ci_widths,
    }


def create_stage_snapshot_figure(stage_name, labels, result, context_text=None):
    _apply_global_plot_style()
    colors = _style()
    player_colors = _player_colors(len(labels))

    win_probs = result["win_prob"]
    tie_probs = result["tie_prob"]
    ci95 = result["win_ci95"]
    uncertainty = stage_uncertainty_stats(result, labels)
    entropy = uncertainty["entropy_bits"]
    entropy_ratio = uncertainty["entropy_ratio"]
    ci_widths = uncertainty["ci_widths"]

    ci_lower = []
    ci_upper = []
    for i, p in enumerate(win_probs):
        lo, hi = ci95[i]
        ci_lower.append(max(0.0, p - lo))
        ci_upper.append(max(0.0, hi - p))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})
    fig.patch.set_facecolor(colors["bg"])
    fig.suptitle(f"{stage_name.upper()} Probability Snapshot", fontsize=14, fontweight="bold")
    if context_text:
        fig.text(
            0.5,
            0.93,
            context_text,
            ha="center",
            va="top",
            fontsize=9,
            color=colors["text_muted"],
        )

    x = list(range(len(labels)))
    bar_width = 0.38

    axes[0].bar(
        [i - bar_width / 2 for i in x],
        win_probs,
        width=bar_width,
        label="P(win)",
        yerr=[ci_lower, ci_upper],
        capsize=4,
        color=colors["win"],
        alpha=0.9,
    )
    axes[0].bar(
        [i + bar_width / 2 for i in x],
        tie_probs,
        width=bar_width,
        label="P(tie)",
        color=colors["tie"],
        alpha=0.8,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Win / Tie Probability (+95% CI on wins)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.4, linestyle="--", color=colors["grid"])
    _beautify_axis(axes[0])

    for i, p in enumerate(win_probs):
        axes[0].text(
            i - bar_width / 2,
            min(0.98, p + 0.03),
            f"{p:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors["text_muted"],
        )

    if sum(win_probs) > 0:
        wedges, texts, autotexts = axes[1].pie(
            win_probs,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=player_colors,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            pctdistance=0.75,
        )
        plt.setp(autotexts, color="white", fontsize=9, fontweight="bold")
        # Donut-style center improves readability for many players.
        centre = plt.Circle((0, 0), 0.52, fc="white")
        axes[1].add_artist(centre)
        axes[1].text(0, 0, f"H={entropy:.2f}\nbits", ha="center", va="center", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "No win probability mass", ha="center", va="center")
    axes[1].set_title(
        f"Win Distribution\nEntropy={entropy:.3f} bits ({entropy_ratio:.1%} of max)"
    )

    y = list(range(len(labels)))
    axes[2].barh(y, ci_widths, color=colors["ci"], alpha=0.85)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(labels, fontsize=9)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("95% CI Width")
    axes[2].set_title("95% CI Uncertainty by Player")
    axes[2].grid(axis="x", alpha=0.4, linestyle="--", color=colors["grid"])
    _beautify_axis(axes[2])
    for i, width in enumerate(ci_widths):
        axes[2].text(min(0.98, width + 0.02), i, f"{width:.3f}", va="center", fontsize=8.5)

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_stage_snapshot(stage_name, labels, result, context_text=None):
    fig = create_stage_snapshot_figure(stage_name, labels, result, context_text=context_text)
    _try_fullscreen(fig)
    plt.show()


def create_challenge_story_figure(stage_records, labels):
    if not stage_records:
        return None

    _apply_global_plot_style()
    colors = _style()
    player_colors = _player_colors(len(labels))

    streets = [r["stage_name"] for r in stage_records]
    x = list(range(len(streets)))
    entropies = _entropy_series(stage_records)
    entropy_max = math.log2(len(labels)) if len(labels) > 1 else 1.0
    normalized_entropy = [e / entropy_max if entropy_max > 0 else 0.0 for e in entropies]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    fig.patch.set_facecolor(colors["bg"])
    fig.suptitle("Texas Hold'em Probability Story", fontsize=15, fontweight="bold")

    for i, label in enumerate(labels):
        y = [record["win_prob"][i] for record in stage_records]
        lo = []
        hi = []
        for record in stage_records:
            p = record["win_prob"][i]
            lo_i, hi_i = record["win_ci95"][i]
            lo.append(lo_i)
            hi.append(hi_i)

        axes[0].plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            markersize=6,
            label=label,
            color=player_colors[i],
        )
        axes[0].fill_between(x, lo, hi, alpha=0.15, color=player_colors[i])

    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("P(win)")
    axes[0].set_title("Equity Timeline with 95% Confidence Bands")
    axes[0].grid(axis="y", alpha=0.35, linestyle="--", color=colors["grid"])
    axes[0].legend(loc="best", ncol=2, frameon=False)
    _beautify_axis(axes[0])

    axes[1].plot(
        x,
        entropies,
        marker="o",
        linewidth=2.5,
        color=colors["entropy"],
        label="Entropy (bits)",
    )
    axes[1].plot(
        x,
        normalized_entropy,
        marker="s",
        linestyle="--",
        linewidth=2,
        color=colors["entropy_norm"],
        label="Normalized entropy",
    )
    axes[1].set_ylabel("Uncertainty")
    axes[1].set_title("Uncertainty Reduction Across Streets")
    axes[1].grid(axis="y", alpha=0.35, linestyle="--", color=colors["grid"])
    axes[1].legend(loc="best", frameon=False)
    _beautify_axis(axes[1])

    stage_labels = []
    for record in stage_records:
        board_text = record.get("board_text", "").strip()
        if board_text:
            stage_labels.append(f"{record['stage_name']}\n{board_text}")
        else:
            stage_labels.append(record["stage_name"])

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stage_labels, fontsize=9)
    axes[1].set_xlabel("Street")
    if entropy_max > 0:
        axes[1].set_ylim(0, max(1.05, entropy_max * 1.05))

    for i, e in enumerate(entropies):
        axes[1].text(
            x[i],
            e + 0.03,
            f"{e:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors["text_muted"],
        )

    plt.tight_layout()
    return fig


def plot_challenge_story(stage_records, labels):
    fig = create_challenge_story_figure(stage_records, labels)
    if fig is None:
        return
    _try_fullscreen(fig)
    plt.show()


def plot_win_probs(win_probs, labels, title="Win Probability by Player"):
    # Backward-compatible helper.
    record = {
        "win_prob": win_probs,
        "tie_prob": [0.0 for _ in labels],
        "win_ci95": [(p, p) for p in win_probs],
    }
    plot_stage_snapshot(stage_name=title, labels=labels, result=record)


def plot_equity_timeline(history, labels):
    # Backward-compatible helper for prior dict format: stage -> win_prob list.
    stage_records = []
    for stage_name, win_prob in history.items():
        stage_records.append(
            {
                "stage_name": stage_name,
                "win_prob": list(win_prob),
                "tie_prob": [0.0 for _ in labels],
                "win_ci95": [(p, p) for p in win_prob],
            }
        )
    plot_challenge_story(stage_records, labels)