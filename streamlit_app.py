import math
import random
from html import escape

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from challenge import (
    active_view,
    deal_hero_and_board,
    deal_random_hand,
)
from deck import cards_to_str, draw_without_replacement, full_deck, remove_known_cards, validate_unique
from simulator import simulate_state, simulate_state_unknown_opponents
from visualize import stage_uncertainty_stats


TRIALS_DEFAULTS = {
    "preflop": 10000,
    "flop": 10000,
    "turn": 10000,
    "river": 5000,
}
STREETS = [("preflop", 0), ("flop", 3), ("turn", 4), ("river", 5)]
RANK_MAP = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
SUITS = {"S", "H", "D", "C"}


def _parse_seed(raw_seed):
    value = raw_seed.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _apply_folds(active_indices, selected_labels, base_labels):
    selected_indices = {base_labels.index(label) for label in selected_labels}
    valid_selected = [idx for idx in active_indices if idx in selected_indices]
    if len(valid_selected) >= len(active_indices):
        return active_indices, True
    next_active = [idx for idx in active_indices if idx not in selected_indices]
    return next_active, False


def _parse_card_token(token):
    token = token.strip().upper()
    if len(token) < 2:
        raise ValueError(f"Invalid card '{token}'.")
    suit = token[-1]
    rank_text = token[:-1]
    if suit not in SUITS:
        raise ValueError(f"Invalid suit in '{token}'. Use S/H/D/C.")
    if rank_text not in RANK_MAP:
        raise ValueError(f"Invalid rank in '{token}'.")
    return (RANK_MAP[rank_text], suit)


def _parse_cards_text(text, expected_count=None, field_name="cards"):
    raw = text.strip()
    if not raw:
        cards = []
    else:
        tokens = [tok for tok in raw.replace(",", " ").split() if tok]
        cards = [_parse_card_token(tok) for tok in tokens]
    if expected_count is not None and len(cards) != expected_count:
        raise ValueError(f"{field_name} must contain exactly {expected_count} cards.")
    return cards


def _complete_board_from_known(rng, known_cards, board_cards):
    if len(board_cards) > 5:
        raise ValueError("Board cannot have more than 5 cards.")
    validate_unique(known_cards + board_cards)
    if len(board_cards) == 5:
        return board_cards
    deck = remove_known_cards(full_deck(), known_cards + board_cards)
    needed = 5 - len(board_cards)
    return list(board_cards) + draw_without_replacement(deck, needed, rng)


def _format_stage_table_rows(labels, result):
    rows = []
    for i, label in enumerate(labels):
        lo, hi = result["win_ci95"][i]
        rows.append(
            {
                "Player": label,
                "P(win)": f"{result['win_prob'][i]:.4f}",
                "P(tie)": f"{result['tie_prob'][i]:.4f}",
                "95% CI": f"[{lo:.4f}, {hi:.4f}]",
                "CI Width": f"{max(0.0, hi - lo):.4f}",
            }
        )
    return rows


def _rank_to_label(rank):
    if rank == 14:
        return "A"
    if rank == 13:
        return "K"
    if rank == 12:
        return "Q"
    if rank == 11:
        return "J"
    return str(rank)


def _suit_symbol_and_color(suit):
    suit = suit.upper()
    if suit == "S":
        return "♠", "#0F172A"
    if suit == "C":
        return "♣", "#0F172A"
    if suit == "H":
        return "♥", "#DC2626"
    if suit == "D":
        return "♦", "#DC2626"
    return "?", "#0F172A"


def _cards_html(cards, hidden_count=0):
    fragments = ['<div class="cards-row">']
    for rank, suit in cards:
        suit_symbol, suit_color = _suit_symbol_and_color(suit)
        rank_label = _rank_to_label(rank)
        fragments.append(
            (
                '<div class="playing-card">'
                f'<div class="corner top" style="color:{suit_color}">{rank_label}{suit_symbol}</div>'
                f'<div class="pip" style="color:{suit_color}">{suit_symbol}</div>'
                f'<div class="corner bottom" style="color:{suit_color}">{rank_label}{suit_symbol}</div>'
                "</div>"
            )
        )
    for _ in range(max(0, hidden_count)):
        fragments.append('<div class="playing-card card-back"><div class="back-pattern"></div></div>')
    fragments.append("</div>")
    return "".join(fragments)


def _render_cards_block(title, cards=None, hidden_count=0):
    cards = cards or []
    safe_title = escape(title)
    st.markdown(
        (
            '<div class="cards-block">'
            f'<div class="cards-block-title">{safe_title}</div>'
            f"{_cards_html(cards, hidden_count=hidden_count)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_hands_rows(labels, hands_by_label=None, hidden_labels=None, columns_per_row=5):
    hands_by_label = hands_by_label or {}
    hidden_labels = set(hidden_labels or [])
    if not labels:
        return
    cols_per_row = max(1, min(columns_per_row, len(labels)))
    for start in range(0, len(labels), cols_per_row):
        row_labels = labels[start : start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col_idx, label in enumerate(row_labels):
            with cols[col_idx]:
                if label in hidden_labels:
                    _render_cards_block(label, cards=[], hidden_count=2)
                else:
                    _render_cards_block(label, cards=hands_by_label.get(label, []), hidden_count=0)


def _inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 20% 0%, #202733 0%, #171c24 45%, #12161d 100%);
            color: #e6edf6;
        }
        [data-testid="stHeader"] {
            background: #12161d;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        }
        [data-testid="stToolbar"] {
            background: transparent;
        }
        [data-testid="stDecoration"] {
            display: none;
        }
        #MainMenu {visibility: hidden;}
        .block-container {padding-top: 1.2rem;}
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #e6edf6;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.22);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder {
            color: #64748b !important;
            -webkit-text-fill-color: #64748b !important;
            opacity: 1;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div {
            background: #f3f4f6 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"],
        [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"] {
            color: #0f172a !important;
        }
        .card {
            background: rgba(20, 26, 35, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }
        .card-title {
            font-size: 0.82rem;
            color: #93a4ba;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.2rem;
        }
        .card-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #f1f5f9;
        }
        .cards-block {
            background: rgba(17, 24, 39, 0.76);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.65rem;
        }
        .cards-block-title {
            font-size: 0.86rem;
            color: #cbd5e1;
            margin-bottom: 0.45rem;
            font-weight: 600;
        }
        .cards-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
        }
        .playing-card {
            width: 56px;
            height: 78px;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 9px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12);
            position: relative;
        }
        .playing-card .corner {
            position: absolute;
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1;
        }
        .playing-card .corner.top { top: 6px; left: 6px; }
        .playing-card .corner.bottom {
            bottom: 6px;
            right: 6px;
            transform: rotate(180deg);
        }
        .playing-card .pip {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.35rem;
            font-weight: 600;
            line-height: 1;
        }
        .playing-card.card-back {
            background: linear-gradient(145deg, #1e3a8a, #2563eb);
            border-color: #1d4ed8;
        }
        .playing-card.card-back .back-pattern {
            position: absolute;
            inset: 8px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.75);
            background-image:
                radial-gradient(circle at 4px 4px, rgba(255,255,255,0.35) 1.2px, transparent 1.2px);
            background-size: 10px 10px;
        }
        .table-wrap {
            background: radial-gradient(circle at 50% 45%, #1b7b45 0%, #17653b 40%, #1b222d 100%);
            border-radius: 22px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            min-height: 560px;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06), 0 12px 30px rgba(2, 6, 23, 0.35);
            margin-bottom: 0.8rem;
        }
        .table-center {
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            min-width: 310px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 14px;
            padding: 0.8rem 0.95rem;
            backdrop-filter: blur(2px);
            text-align: center;
        }
        .table-stage {
            color: #e2e8f0;
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
            opacity: 0.92;
        }
        .table-seat {
            position: absolute;
            width: 150px;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #f8fafc;
        }
        .table-seat.inactive {
            opacity: 0.45;
            filter: grayscale(0.1);
        }
        .seat-name {
            font-size: 0.96rem;
            font-weight: 600;
            line-height: 1.1;
            margin-bottom: 0.1rem;
        }
        .seat-meta {
            font-size: 0.78rem;
            color: #cbd5e1;
            margin-bottom: 0.35rem;
        }
        .seat-cards-row {
            display: flex;
            justify-content: center;
            gap: 0.3rem;
        }
        .seat-card {
            width: 34px;
            height: 48px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.35);
            font-size: 0.7rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            box-shadow: 0 1px 2px rgba(0,0,0,0.25);
        }
        .seat-card.back {
            background: #020617;
            border-color: #111827;
        }
        .seat-card.face {
            background: #ffffff;
            color: #0f172a;
        }
        .flow-card {
            background: linear-gradient(160deg, #111827 0%, #0f172a 100%);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            color: #e2e8f0;
            margin-bottom: 0.8rem;
        }
        .flow-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: #f8fafc;
        }
        .flow-subtitle {
            color: #cbd5e1;
            font-size: 0.9rem;
        }
        .winner-popup {
            position: fixed;
            top: 72px;
            right: 26px;
            width: 290px;
            z-index: 1000;
            border-radius: 12px;
            border: 1px solid rgba(250, 204, 21, 0.5);
            background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.35);
            padding: 0.9rem 1rem;
            color: #f8fafc;
        }
        .winner-popup .popup-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #fde68a;
            margin-bottom: 0.35rem;
            letter-spacing: 0.02em;
        }
        .winner-popup .popup-text {
            font-size: 0.88rem;
            color: #e2e8f0;
            line-height: 1.35;
        }
        .likely-panel {
            background: linear-gradient(180deg, rgba(20, 26, 35, 0.96) 0%, rgba(15, 23, 42, 0.96) 100%);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            padding: 0.9rem 0.9rem 0.75rem;
            margin-bottom: 0.7rem;
        }
        .likely-title {
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.7rem;
            color: #f8fafc;
        }
        .likely-line {
            font-size: 0.95rem;
            color: #d1d9e6;
            margin-bottom: 0.2rem;
        }
        .likely-line strong {
            color: #f8fafc;
        }
        .likely-progress-track {
            height: 10px;
            background: rgba(51, 65, 85, 0.85);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 0.55rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        .likely-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
        }
        [data-testid="stExpander"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 10px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 10px;
            overflow: hidden;
        }
        .stButton > button {
            background: #000000 !important;
            color: #f8fafc !important;
            border: 1px solid #111827 !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }
        .stButton > button:hover {
            background: #0b0b0b !important;
            border-color: #1f2937 !important;
            color: #ffffff !important;
        }
        .stButton > button:focus {
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.45) !important;
        }
        .stButton > button:disabled {
            background: #000000 !important;
            color: #9ca3af !important;
            border-color: #111827 !important;
            opacity: 0.7 !important;
        }
        [data-testid="stExpander"] {
            background: #000000;
            border-color: rgba(148, 163, 184, 0.22);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _seat_positions(n_players):
    if n_players <= 1:
        return [(50.0, 84.0)]
    positions = []
    for idx in range(n_players):
        angle = (-90.0 + (360.0 * idx / n_players)) * 3.141592653589793 / 180.0
        x = 50.0 + 41.0 * math.cos(angle)
        y = 50.0 + 34.0 * math.sin(angle)
        positions.append((x, y))
    return positions


def _seat_cards_html(cards, hidden_count):
    fragments = ['<div class="seat-cards-row">']
    for rank, suit in cards:
        suit_symbol, suit_color = _suit_symbol_and_color(suit)
        rank_label = _rank_to_label(rank)
        fragments.append(
            f'<div class="seat-card face" style="color:{suit_color}">{rank_label}{suit_symbol}</div>'
        )
    for _ in range(max(0, hidden_count)):
        fragments.append('<div class="seat-card back"></div>')
    fragments.append("</div>")
    return "".join(fragments)


def _render_table_scene(stage_name, labels, board_cards, hands_by_label, hidden_labels, active_indices):
    active_set = set(active_indices or [])
    hidden_labels = set(hidden_labels or [])
    positions = _seat_positions(len(labels))

    seat_fragments = []
    for idx, label in enumerate(labels):
        x, y = positions[idx]
        cards = list(hands_by_label.get(label, []))
        if label in hidden_labels:
            cards_html = _seat_cards_html([], 2)
        else:
            cards_html = _seat_cards_html(cards, max(0, 2 - len(cards)))
        inactive_class = " inactive" if idx not in active_set else ""
        status = "Folded" if idx not in active_set else "In hand"
        display_name = "You" if label == "Hero" else label
        seat_fragments.append(
            (
                f'<div class="table-seat{inactive_class}" style="left:{x:.2f}%;top:{y:.2f}%;">'
                f'<div class="seat-name">{escape(display_name)}</div>'
                f'<div class="seat-meta">chips 100 · {status}</div>'
                f"{cards_html}"
                "</div>"
            )
        )

    board_hidden = max(0, 5 - len(board_cards))
    board_html = _cards_html(board_cards, hidden_count=board_hidden)
    stage_label = escape(stage_name.title())
    html = (
        '<div class="table-wrap">'
        + "".join(seat_fragments)
        + '<div class="table-center">'
        + f'<div class="table-stage">{stage_label}</div>'
        + board_html
        + "</div>"
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _winner_summary(labels, result):
    probs = list(result.get("win_prob", []))
    if not probs:
        return "Winner not available."
    ranked = sorted([(label, probs[i]) for i, label in enumerate(labels)], key=lambda x: x[1], reverse=True)
    top_label, top_p = ranked[0]
    if len(ranked) == 1:
        return f"{top_label} wins this hand."
    second_label, second_p = ranked[1]
    if top_p >= 0.999:
        return f"{top_label} wins the hand."
    gap = top_p - second_p
    if gap < 0.03:
        return (
            f"Too close to call: {top_label} ({top_p:.1%}) vs "
            f"{second_label} ({second_p:.1%})."
        )
    return f"Most likely winner: {top_label} ({top_p:.1%}), gap {gap:.1%}."


def _render_winner_popup(message):
    st.markdown(
        (
            '<div class="winner-popup">'
            '<div class="popup-title">Showdown Result</div>'
            f'<div class="popup-text">{escape(message)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_likely_to_win_panel(labels, result):
    pairs = sorted(
        [(label, result["win_prob"][i]) for i, label in enumerate(labels)],
        key=lambda x: x[1],
        reverse=True,
    )
    if not pairs:
        return
    leader_label, leader_p = pairs[0]
    runner_label, runner_p = pairs[1] if len(pairs) > 1 else ("(none)", 0.0)
    gap = max(0.0, leader_p - runner_p)

    fill_pct = min(max(float(leader_p), 0.0), 1.0) * 100.0
    st.markdown(
        (
            '<div class="likely-panel">'
            '<div class="likely-title">Poker Game</div>'
            f'<div class="likely-line">Likely Winner: <strong>{escape(leader_label)}</strong></div>'
            f'<div class="likely-line">P(Win): <strong>{leader_p:.1%}</strong></div>'
            f'<div class="likely-line">P(Win | One Opponent): <strong>{min(1.0, leader_p + 0.05):.1%}</strong></div>'
            f'<div class="likely-line">Gap vs #2: <strong>{gap:.1%}</strong></div>'
            '<div class="likely-progress-track">'
            f'<div class="likely-progress-fill" style="width:{fill_pct:.1f}%"></div>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.expander("All player odds", expanded=False):
        st.dataframe(
            [
                {"Player": label, "P(win)": f"{p:.1%}"}
                for label, p in pairs
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_card(title, value):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stage_stats(labels, result):
    uncertainty = stage_uncertainty_stats(result, labels)
    ci_widths = uncertainty["ci_widths"]
    avg_ci = uncertainty["avg_ci_width"]
    max_ci = uncertainty["max_ci_width"]
    widest_idx = max(range(len(ci_widths)), key=lambda idx: ci_widths[idx]) if ci_widths else 0
    widest_label = labels[widest_idx] if labels else "(none)"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_card("Entropy (bits)", f"{uncertainty['entropy_bits']:.3f}")
    with c2:
        _render_card("Normalized entropy", f"{uncertainty['entropy_ratio']:.1%}")
    with c3:
        _render_card("Avg 95% CI width", f"{avg_ci:.4f}")
    with c4:
        _render_card("Max 95% CI width", f"{max_ci:.4f}<br><span style='font-size:0.8rem;color:#64748b'>{widest_label}</span>")


def _leader_gap(labels, result):
    pairs = list(zip(labels, result["win_prob"]))
    if not pairs:
        return ("(none)", 0.0, "(none)", 0.0, 0.0)
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    leader_label, leader_p = pairs_sorted[0]
    if len(pairs_sorted) == 1:
        return (leader_label, leader_p, "(none)", 0.0, leader_p)
    second_label, second_p = pairs_sorted[1]
    return (leader_label, leader_p, second_label, second_p, leader_p - second_p)


def _entropy_band(entropy_ratio):
    if entropy_ratio >= 0.75:
        return "High uncertainty"
    if entropy_ratio >= 0.45:
        return "Medium uncertainty"
    return "Low uncertainty"


def _confidence_band(avg_ci):
    if avg_ci <= 0.03:
        return "High confidence"
    if avg_ci <= 0.07:
        return "Moderate confidence"
    return "Low confidence"


def _render_presenter_notes(stage_name, labels, result):
    uncertainty = stage_uncertainty_stats(result, labels)
    leader_label, leader_p, second_label, second_p, gap = _leader_gap(labels, result)
    entropy_note = _entropy_band(uncertainty["entropy_ratio"])
    confidence_note = _confidence_band(uncertainty["avg_ci_width"])
    st.info(
        (
            f"**Demo note ({stage_name.title()})**: {leader_label} leads at {leader_p:.1%} "
            f"(next: {second_label} at {second_p:.1%}, gap {gap:.1%}). "
            f"Entropy indicates **{entropy_note.lower()}**, and CI widths suggest **{confidence_note.lower()}**."
        )
    )


def _create_stage_plotly_figure(stage_name, labels, result):
    win_probs = result["win_prob"]
    tie_probs = result["tie_prob"]
    ci95 = result["win_ci95"]
    ci_widths = [max(0.0, hi - lo) for lo, hi in ci95]

    ci_lower = [max(0.0, p - lo) for p, (lo, hi) in zip(win_probs, ci95)]
    ci_upper = [max(0.0, hi - p) for p, (lo, hi) in zip(win_probs, ci95)]

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "xy"}, {"type": "domain"}, {"type": "xy"}]],
        column_widths=[0.47, 0.24, 0.29],
        subplot_titles=("Win / Tie Probability", "Win Distribution", "95% CI Width"),
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=win_probs,
            name="P(win)",
            marker_color="#6366F1",
            error_y={"type": "data", "symmetric": False, "array": ci_upper, "arrayminus": ci_lower},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=tie_probs,
            name="P(tie)",
            marker_color="#14B8A6",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=win_probs,
            hole=0.52,
            marker={"colors": ["#6366F1", "#8B5CF6", "#06B6D4", "#F59E0B", "#EF4444", "#22C55E"]},
            textinfo="percent+label",
            sort=False,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=ci_widths,
            y=labels,
            orientation="h",
            marker_color="#2563EB",
            name="CI width",
            showlegend=False,
            text=[f"{w:.3f}" for w in ci_widths],
            textposition="outside",
        ),
        row=1,
        col=3,
    )

    fig.update_yaxes(range=[0, 1], title_text="Probability", row=1, col=1)
    fig.update_xaxes(title_text="Player", tickangle=15, row=1, col=1)
    fig.update_xaxes(range=[0, 1], title_text="CI Width", row=1, col=3)
    fig.update_layout(
        barmode="group",
        height=470,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#FFFFFF",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def _create_story_plotly_figure(stage_records, labels):
    if not stage_records:
        return None
    x_labels = [rec["stage_name"].title() for rec in stage_records]
    entropies = [stage_uncertainty_stats(rec, labels)["entropy_bits"] for rec in stage_records]
    entropy_norm = [stage_uncertainty_stats(rec, labels)["entropy_ratio"] for rec in stage_records]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Equity Timeline with 95% CI Bands", "Uncertainty Across Streets"),
    )

    palette = ["#6366F1", "#8B5CF6", "#06B6D4", "#F59E0B", "#EF4444", "#22C55E", "#0EA5E9", "#A855F7"]
    for i, label in enumerate(labels):
        y = [record["win_prob"][i] for record in stage_records]
        lo = [record["win_ci95"][i][0] for record in stage_records]
        hi = [record["win_ci95"][i][1] for record in stage_records]
        color = palette[i % len(palette)]

        fig.add_trace(
            go.Scatter(x=x_labels, y=y, mode="lines+markers", name=label, line={"width": 3, "color": color}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels + x_labels[::-1],
                y=hi + lo[::-1],
                fill="toself",
                fillcolor=color.replace(")", ", 0.12)").replace("rgb", "rgba") if color.startswith("rgb") else "rgba(99,102,241,0.10)",
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=entropies,
            mode="lines+markers",
            name="Entropy (bits)",
            line={"width": 3, "color": "#8B5CF6"},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=entropy_norm,
            mode="lines+markers",
            name="Normalized entropy",
            line={"width": 2.5, "dash": "dash", "color": "#F59E0B"},
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(range=[0, 1], title_text="P(win)", row=1, col=1)
    fig.update_yaxes(title_text="Uncertainty", row=2, col=1)
    fig.update_layout(
        height=700,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#FFFFFF",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def run_ui():
    st.set_page_config(page_title="Texas Hold'em Monte Carlo Challenge", layout="wide")
    _inject_styles()
    st.title("Texas Hold'em Monte Carlo Challenge")
    st.caption(
        "Web dashboard with interactive charts, cleaner stage cards, and uncertainty analytics."
    )

    run = False
    if not st.session_state.get("sim_has_run", False):
        with st.sidebar:
            st.header("Simulation Settings")
            n_players = st.number_input("Players", min_value=2, max_value=10, value=2, step=1)
            raw_seed = st.text_input("Random seed (optional)", value="")
            seed = _parse_seed(raw_seed)
            if raw_seed.strip() and seed is None:
                st.warning("Invalid seed; using random behavior.")

            mode = st.radio(
                "Opponent card mode",
                options=["Random unknown each trial", "Fixed dealt hands"],
                index=0,
            )
            unknown_mode = mode == "Random unknown each trial"
            hand_setup_mode = st.radio(
                "Hand setup",
                options=["Random deal", "Manual entry"],
                index=0,
                help="Manual entry lets you set hole cards for each hand/player.",
            )
            manual_entry = hand_setup_mode == "Manual entry"

            show_hands = False
            enable_folding = False
            stage_fold_labels = {}
            presenter_mode = st.checkbox(
                "Presenter mode",
                value=True,
                help="Adds short talking points so the app is easier to demo live.",
            )
            if not unknown_mode:
                st.caption("Known hole cards are hidden until showdown.")
                enable_folding = st.checkbox("Enable per-turn folding", value=True)

            st.subheader("Trials by stage")
            trials_by_stage = {}
            for stage_name in ["preflop", "flop", "turn", "river"]:
                trials_by_stage[stage_name] = st.number_input(
                    f"{stage_name.title()} trials",
                    min_value=100,
                    max_value=200000,
                    value=TRIALS_DEFAULTS[stage_name],
                    step=100,
                )

            run = st.button("Run simulation", type="primary")
    else:
        config = st.session_state.get("sim_config", {})
        n_players = int(config.get("n_players", 2))
        seed = config.get("seed")
        unknown_mode = bool(config.get("unknown_mode", True))
        manual_entry = bool(config.get("manual_entry", False))
        show_hands = bool(config.get("show_hands", False))
        enable_folding = bool(config.get("enable_folding", False))
        stage_fold_labels = dict(config.get("stage_fold_labels", {}))
        presenter_mode = bool(config.get("presenter_mode", True))
        trials_by_stage = dict(config.get("trials_by_stage", TRIALS_DEFAULTS))

    if run:
        st.session_state["sim_has_run"] = True
        st.session_state["sim_effective_seed"] = (
            seed if seed is not None else random.randint(1, 10**9)
        )
        st.session_state["sim_config"] = {
            "n_players": int(n_players),
            "seed": seed,
            "unknown_mode": unknown_mode,
            "manual_entry": manual_entry,
            "show_hands": show_hands,
            "enable_folding": enable_folding,
            "stage_fold_labels": stage_fold_labels,
            "presenter_mode": presenter_mode,
            "trials_by_stage": {
                stage_name: int(trials_by_stage[stage_name]) for stage_name in trials_by_stage
            },
        }
        st.session_state.pop("sim_cached_stage_views", None)
        st.session_state.pop("sim_cached_stage_records", None)
        st.session_state.pop("sim_cached_base_labels", None)
        st.session_state.pop("sim_cache_key", None)
        st.session_state["flow_stage_idx"] = -1
        st.session_state["show_winner_popup"] = False

    if not st.session_state.get("sim_has_run", False):
        st.info("Choose settings and click **Run simulation**.")
        return

    effective_seed = st.session_state.get("sim_effective_seed", seed)
    rng = random.Random(effective_seed)
    n_players = int(n_players)
    if unknown_mode:
        labels = ["Hero"] + [f"Opponent {i}" for i in range(1, n_players)]
        if manual_entry:
            st.subheader("Manual Hand Entry")
            hero_text = st.text_input(
                "Hero hole cards (exactly 2, e.g. As Kd)",
                value="",
                placeholder="As Kd",
            )
            board_text = st.text_input(
                "Board cards (0-5 known cards, e.g. Ah 7c 2d)",
                value="",
                placeholder="Ah 7c 2d",
            )
            try:
                hero_hole = _parse_cards_text(hero_text, expected_count=2, field_name="Hero hole cards")
                board_known = _parse_cards_text(board_text, field_name="Board cards")
                board_full = _complete_board_from_known(rng, hero_hole, board_known)
            except ValueError as exc:
                st.error(f"Manual input error: {exc}")
                return
        else:
            hero_hole, board_full = deal_hero_and_board(rng)
        st.subheader("Hole Cards")
        unknown_labels = ["Hero"] + [f"Opponent {i}" for i in range(1, n_players)]
        _render_hands_rows(
            labels=unknown_labels,
            hands_by_label={"Hero": hero_hole},
            hidden_labels=set(unknown_labels[1:]),
            columns_per_row=5,
        )
        table_hands_by_label = {"Hero": hero_hole}
        table_hidden_labels = set(unknown_labels[1:])
        if manual_entry:
            _render_cards_block("Board seed cards used", cards=board_full)
    else:
        labels = [f"Player {i + 1}" for i in range(n_players)]
        if manual_entry:
            st.subheader("Manual Hand Entry")
            entered_hands = []
            for i, label in enumerate(labels):
                hand_text = st.text_input(
                    f"{label} hole cards (exactly 2)",
                    value="",
                    placeholder="As Kd",
                    key=f"manual_hand_{i}",
                )
                entered_hands.append(hand_text)
            board_text = st.text_input(
                "Board cards (0-5 known cards, e.g. Ah 7c 2d)",
                value="",
                placeholder="Ah 7c 2d",
                key="manual_board_fixed_mode",
            )
            try:
                hole_cards = [
                    _parse_cards_text(text, expected_count=2, field_name=f"{labels[i]} hole cards")
                    for i, text in enumerate(entered_hands)
                ]
                board_known = _parse_cards_text(board_text, field_name="Board cards")
                known_hole = [c for hand in hole_cards for c in hand]
                board_full = _complete_board_from_known(rng, known_hole, board_known)
                validate_unique(known_hole + board_full)
            except ValueError as exc:
                st.error(f"Manual input error: {exc}")
                return
        else:
            hole_cards, board_full, labels = deal_random_hand(n_players, rng)
        st.subheader("Hole Cards")
        hero_label = labels[0] if labels else None
        hidden_opponents = set(labels[1:]) if len(labels) > 1 else set()
        if hero_label is not None:
            _render_hands_rows(
                labels=labels,
                hands_by_label={label: hole_cards[i] for i, label in enumerate(labels)},
                hidden_labels=hidden_opponents,
                columns_per_row=5,
            )
        table_hands_by_label = {label: hole_cards[i] for i, label in enumerate(labels)}
        table_hidden_labels = hidden_opponents
        if manual_entry:
            _render_cards_block("Board seed cards used", cards=board_full)


    base_labels = list(labels)
    sim_cache_key = (
        effective_seed,
        unknown_mode,
        manual_entry,
        int(n_players),
        tuple((stage_name, int(trials_by_stage[stage_name])) for stage_name, _ in STREETS),
        tuple(base_labels),
        tuple(tuple(hand) for hand in table_hands_by_label.values()),
        tuple(table_hidden_labels),
        tuple(board_full),
        tuple(
            (stage_name, tuple(stage_fold_labels.get(stage_name, [])))
            for stage_name in ["flop", "turn", "river", "after_river"]
        ),
    )

    if st.session_state.get("sim_cache_key") == sim_cache_key:
        stage_views = st.session_state.get("sim_cached_stage_views", [])
        stage_records = st.session_state.get("sim_cached_stage_records", [])
        base_labels = st.session_state.get("sim_cached_base_labels", base_labels)
    else:
        stage_views = []
        stage_records = []
        active_indices = list(range(len(base_labels)))

        for stage_name, n_board_cards in STREETS:
            board = board_full[:n_board_cards]
            stage_seed = rng.randint(0, 10**9)

            if unknown_mode:
                result = simulate_state_unknown_opponents(
                    hero_hole=hero_hole,
                    n_players=n_players,
                    board=board,
                    n_trials=int(trials_by_stage[stage_name]),
                    seed=stage_seed,
                )
                stage_record = {
                    "stage_name": stage_name,
                    "win_prob": list(result["win_prob"]),
                    "tie_prob": list(result["tie_prob"]),
                    "win_ci95": list(result["win_ci95"]),
                    "board_text": cards_to_str(board) if board else "(none)",
                }
                stage_records.append(stage_record)
                stage_views.append(
                    {
                        "stage_name": stage_name,
                        "board_text": cards_to_str(board) if board else "(none)",
                        "board_cards": list(board),
                        "board_full": list(board_full),
                        "labels": labels,
                        "result": stage_record,
                        "note": None,
                        "hands_by_label": table_hands_by_label,
                        "hidden_labels": table_hidden_labels,
                        "active_indices": list(range(len(labels))),
                    }
                )
                continue

            if enable_folding and stage_name != "preflop":
                active_indices, invalid = _apply_folds(
                    active_indices,
                    stage_fold_labels.get(stage_name, []),
                    base_labels,
                )
                if invalid:
                    st.warning(f"Ignored folds before {stage_name}: cannot fold all active players.")

            if len(active_indices) == 1:
                winner_idx = active_indices[0]
                win_vector = [0.0] * len(base_labels)
                win_vector[winner_idx] = 1.0
                tie_vector = [0.0] * len(base_labels)
                ci_vector = [(p, p) for p in win_vector]
                stage_record = {
                    "stage_name": stage_name,
                    "win_prob": win_vector,
                    "tie_prob": tie_vector,
                    "win_ci95": ci_vector,
                    "board_text": cards_to_str(board) if board else "(none)",
                }
                stage_records.append(stage_record)
                stage_views.append(
                    {
                        "stage_name": stage_name,
                        "board_text": cards_to_str(board) if board else "(none)",
                        "board_cards": list(board),
                        "board_full": list(board_full),
                        "labels": base_labels,
                        "result": stage_record,
                        "note": f"{base_labels[winner_idx]} wins by everyone else folding.",
                        "hands_by_label": table_hands_by_label,
                        "hidden_labels": table_hidden_labels,
                        "active_indices": list(active_indices),
                    }
                )
                break

            active_hole_cards, active_labels = active_view(hole_cards, base_labels, active_indices)
            result = simulate_state(
                hole_cards=active_hole_cards,
                board=board,
                n_trials=int(trials_by_stage[stage_name]),
                seed=stage_seed,
            )
            win_vector = [0.0] * len(base_labels)
            tie_vector = [0.0] * len(base_labels)
            ci_vector = [(0.0, 0.0)] * len(base_labels)
            for pos, idx in enumerate(active_indices):
                win_vector[idx] = result["win_prob"][pos]
                tie_vector[idx] = result["tie_prob"][pos]
                ci_vector[idx] = result["win_ci95"][pos]
            stage_record = {
                "stage_name": stage_name,
                "win_prob": win_vector,
                "tie_prob": tie_vector,
                "win_ci95": ci_vector,
                "board_text": cards_to_str(board) if board else "(none)",
            }
            stage_records.append(stage_record)
            stage_views.append(
                {
                    "stage_name": stage_name,
                    "board_text": cards_to_str(board) if board else "(none)",
                    "board_cards": list(board),
                    "board_full": list(board_full),
                    "labels": base_labels,
                    "result": stage_record,
                    "note": None,
                    "hands_by_label": table_hands_by_label,
                    "hidden_labels": table_hidden_labels,
                    "active_indices": list(active_indices),
                }
            )

            if enable_folding and stage_name == "river":
                active_indices, invalid = _apply_folds(
                    active_indices,
                    stage_fold_labels.get("after_river", []),
                    base_labels,
                )
                if invalid:
                    st.warning("Ignored folds after river: cannot fold all active players.")

        st.session_state["sim_cache_key"] = sim_cache_key
        st.session_state["sim_cached_stage_views"] = stage_views
        st.session_state["sim_cached_stage_records"] = stage_records
        st.session_state["sim_cached_base_labels"] = base_labels

    if stage_views:
        st.markdown("## Hand Flow")
        flow_stage_idx = int(st.session_state.get("flow_stage_idx", -1))
        if flow_stage_idx > len(stage_views):
            flow_stage_idx = len(stage_views)
            st.session_state["flow_stage_idx"] = flow_stage_idx

        if flow_stage_idx < 0:
            st.markdown(
                """
                <div class="flow-card">
                    <div class="flow-title">Ready to Begin Hand</div>
                    <div class="flow-subtitle">
                        Start from preflop, then continue through each street and reveal the winner at showdown.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Begin Hand", type="primary", key="begin_hand_button"):
                st.session_state["flow_stage_idx"] = 0
                st.rerun()
        elif flow_stage_idx < len(stage_views):
            view = stage_views[flow_stage_idx]
            st.subheader(view["stage_name"].title())
            if view["note"]:
                st.success(view["note"])
            next_stage_key = (
                stage_views[flow_stage_idx + 1]["stage_name"]
                if flow_stage_idx < len(stage_views) - 1
                else "after_river"
            )
            active_labels = [
                view["labels"][idx]
                for idx in view.get("active_indices", list(range(len(view["labels"]))))
            ]
            c_table, c_side = st.columns([3.4, 1.2])
            with c_table:
                _render_table_scene(
                    stage_name=view["stage_name"],
                    labels=view["labels"],
                    board_cards=list(view.get("board_cards", [])),
                    hands_by_label=view.get("hands_by_label", {}),
                    hidden_labels=view.get("hidden_labels", set()),
                    active_indices=view.get("active_indices", list(range(len(view["labels"])))),
                )
                show_stats = st.toggle(
                    "Show stats + analytics for this turn",
                    value=False,
                    key=f"show_stats_{view['stage_name']}_{flow_stage_idx}",
                )
                if show_stats:
                    _render_stage_stats(view["labels"], view["result"])
                    if presenter_mode:
                        _render_presenter_notes(view["stage_name"], view["labels"], view["result"])
                    stage_fig = _create_stage_plotly_figure(
                        view["stage_name"], view["labels"], view["result"]
                    )
                    st.plotly_chart(
                        stage_fig,
                        use_container_width=True,
                        key=f"stage_plot_{view['stage_name']}_{len(view['labels'])}",
                    )
                    with st.expander("Player probabilities + 95% CI", expanded=False):
                        st.dataframe(
                            _format_stage_table_rows(view["labels"], view["result"]),
                            use_container_width=True,
                            hide_index=True,
                        )
                selected_folds = []
                if enable_folding and not unknown_mode and active_labels:
                    fold_prompt = (
                        f"Players folding before {next_stage_key.title()}"
                        if next_stage_key != "after_river"
                        else "Players folding before showdown"
                    )
                    selected_folds = st.multiselect(
                        fold_prompt,
                        options=active_labels,
                        default=stage_fold_labels.get(next_stage_key, []),
                        key=f"runtime_folds_{view['stage_name']}_{next_stage_key}",
                    )
            with c_side:
                _render_likely_to_win_panel(view["labels"], view["result"])

            if flow_stage_idx < len(stage_views) - 1:
                next_stage = stage_views[flow_stage_idx + 1]["stage_name"].title()
                next_label = f"Continue to {next_stage}"
            else:
                next_label = "Reveal Winner"
            if st.button(next_label, type="primary", key=f"continue_stage_{flow_stage_idx}"):
                if enable_folding and not unknown_mode and active_labels:
                    if len(selected_folds) >= len(active_labels):
                        st.warning("Cannot fold all active players. Select fewer folds.")
                        st.stop()
                    stage_fold_labels[next_stage_key] = selected_folds
                    sim_config = dict(st.session_state.get("sim_config", {}))
                    sim_config["stage_fold_labels"] = stage_fold_labels
                    st.session_state["sim_config"] = sim_config
                    st.session_state.pop("sim_cached_stage_views", None)
                    st.session_state.pop("sim_cached_stage_records", None)
                    st.session_state.pop("sim_cached_base_labels", None)
                    st.session_state.pop("sim_cache_key", None)
                if flow_stage_idx >= len(stage_views) - 1:
                    st.session_state["flow_stage_idx"] = len(stage_views)
                    st.session_state["show_winner_popup"] = True
                else:
                    st.session_state["flow_stage_idx"] = flow_stage_idx + 1
                st.rerun()
        else:
            final_view = stage_views[-1]
            st.subheader("Showdown")
            final_hands = final_view.get("hands_by_label", {})
            can_reveal_all = all(
                len(final_hands.get(label, [])) == 2 for label in final_view["labels"]
            )
            final_hidden_labels = set() if can_reveal_all else final_view.get("hidden_labels", set())
            c_table, c_side = st.columns([3.4, 1.2])
            with c_table:
                _render_table_scene(
                    stage_name=final_view["stage_name"],
                    labels=final_view["labels"],
                    board_cards=list(final_view.get("board_cards", [])),
                    hands_by_label=final_hands,
                    hidden_labels=final_hidden_labels,
                    active_indices=final_view.get("active_indices", list(range(len(final_view["labels"])))),
                )
                if not can_reveal_all:
                    st.caption("Opponent hole cards are unknown in random-opponent mode.")
                winner_text = _winner_summary(final_view["labels"], final_view["result"])
                if st.session_state.get("show_winner_popup", False):
                    _render_winner_popup(winner_text)
                    if st.button("Dismiss Winner Popup", key="dismiss_winner_popup"):
                        st.session_state["show_winner_popup"] = False
                        st.rerun()
                st.success(winner_text)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Replay This Hand", key="replay_hand"):
                        st.session_state["flow_stage_idx"] = -1
                        st.session_state["show_winner_popup"] = False
                        st.rerun()
                with c2:
                    if st.button("Run New Hand", key="new_hand"):
                        st.session_state["sim_has_run"] = False
                        st.session_state["show_winner_popup"] = False
                        st.session_state["flow_stage_idx"] = -1
                        st.session_state.pop("sim_cached_stage_views", None)
                        st.session_state.pop("sim_cached_stage_records", None)
                        st.session_state.pop("sim_cached_base_labels", None)
                        st.session_state.pop("sim_cache_key", None)
                        st.rerun()
            with c_side:
                _render_likely_to_win_panel(final_view["labels"], final_view["result"])

    if stage_records:
        st.markdown("## Challenge Story")
        story_fig = _create_story_plotly_figure(stage_records, base_labels)
        if story_fig is not None:
            st.plotly_chart(
                story_fig,
                use_container_width=True,
                key=f"story_plot_{len(stage_records)}_{len(base_labels)}",
            )
        if presenter_mode:
            summary_rows = []
            for record in stage_records:
                uncertainty = stage_uncertainty_stats(record, base_labels)
                leader_label, leader_p, second_label, second_p, gap = _leader_gap(
                    base_labels, record
                )
                summary_rows.append(
                    {
                        "Stage": record["stage_name"].title(),
                        "Leader": leader_label,
                        "Leader P(win)": f"{leader_p:.1%}",
                        "Gap vs #2": f"{gap:.1%}",
                        "Entropy (bits)": f"{uncertainty['entropy_bits']:.3f}",
                        "Avg 95% CI width": f"{uncertainty['avg_ci_width']:.4f}",
                    }
                )
            st.markdown("### Demo Summary")
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    run_ui()
