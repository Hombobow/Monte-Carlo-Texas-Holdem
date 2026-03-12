from typing import List, Tuple, Dict
import math
import random

from deck import full_deck, remove_known_cards, validate_unique, draw_without_replacement
from evaluator import rank_7, hand_category_name

Card = Tuple[int, str]


def ci95(p: float, n: int) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    # Binomial standard error for Monte Carlo win-rate estimate p.
    se = math.sqrt(max(p * (1 - p), 0.0) / n)
    # Normal-approximation 95% confidence interval: p ± 1.96 * SE.
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return (lo, hi)


def simulate_state(
    hole_cards: List[List[Card]],
    board: List[Card],
    n_trials: int = 20000,
    seed: int | None = None,
) -> Dict:
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0.")
    if not hole_cards or any(len(h) != 2 for h in hole_cards):
        raise ValueError("Each player must have exactly 2 hole cards.")
    if len(board) > 5:
        raise ValueError("Board cannot exceed 5 cards.")

    known = [c for h in hole_cards for c in h] + list(board)
    validate_unique(known)

    deck = remove_known_cards(full_deck(), known)
    missing = 5 - len(board)
    n_players = len(hole_cards)

    wins = [0] * n_players # wins is a list of 0s, one for each player
    ties = [0] * n_players # ties is a list of 0s, one for each player
    best_categories = [0] * n_players  # from final trial, for quick display only 

    rng = random.Random(seed)

    for _ in range(n_trials):
        future = draw_without_replacement(deck, missing, rng)
        full_board = list(board) + future # full_board is the board + the future cards

        ranks = [] # ranks is a list of ranks for each player
        for i in range(n_players):
            r = rank_7(hole_cards[i] + full_board)
            ranks.append(r)
            best_categories[i] = r[0] # best_categories is a list of the best category for each player

        best = max(ranks) # Picks the strongest rank tuple in this trial.
        winners = [i for i, r in enumerate(ranks) if r == best] # winners is a list of the indices of the players with the best rank

        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            for w in winners:
                ties[w] += 1

    win_prob = [w / n_trials for w in wins]
    tie_prob = [t / n_trials for t in ties]
    # Compute per-player 95% CI around each estimated win probability.
    ci = [ci95(p, n_trials) for p in win_prob]
    category_names = [hand_category_name((c,)) for c in best_categories]

    return {
        "win_prob": win_prob,
        "tie_prob": tie_prob,
        "wins": wins,
        "ties": ties,
        "n_trials": n_trials,
        "win_ci95": ci,
        "category_hint": category_names,
    }


def simulate_state_unknown_opponents(
    hero_hole: List[Card],
    n_players: int,
    board: List[Card],
    n_trials: int = 200000,
    seed: int | None = None,
) -> Dict:
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0.")
    if len(hero_hole) != 2:
        raise ValueError("Hero must have exactly 2 hole cards.")
    if n_players < 2 or n_players > 10:
        raise ValueError("n_players must be between 2 and 10.")
    if len(board) > 5:
        raise ValueError("Board cannot exceed 5 cards.")

    known = list(hero_hole) + list(board)
    validate_unique(known)

    deck = remove_known_cards(full_deck(), known)
    missing_board = 5 - len(board)
    opp_count = n_players - 1
    cards_per_trial = missing_board + (2 * opp_count)

    wins = [0] * n_players
    ties = [0] * n_players
    best_categories = [0] * n_players

    rng = random.Random(seed)

    for _ in range(n_trials):
        sampled = draw_without_replacement(deck, cards_per_trial, rng)
        future_board = sampled[:missing_board]
        opp_cards_flat = sampled[missing_board:]
        full_board = list(board) + future_board

        hole_cards = [list(hero_hole)]
        for i in range(opp_count):
            c1 = opp_cards_flat[2 * i]
            c2 = opp_cards_flat[2 * i + 1]
            hole_cards.append([c1, c2])

        ranks = []
        for i in range(n_players):
            r = rank_7(hole_cards[i] + full_board)
            ranks.append(r)
            best_categories[i] = r[0]

        best = max(ranks)
        winners = [i for i, r in enumerate(ranks) if r == best]

        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            for w in winners:
                ties[w] += 1

    win_prob = [w / n_trials for w in wins]
    tie_prob = [t / n_trials for t in ties]
    # Compute per-player 95% CI around each estimated win probability.
    ci = [ci95(p, n_trials) for p in win_prob]
    category_names = [hand_category_name((c,)) for c in best_categories]

    return {
        "win_prob": win_prob,
        "tie_prob": tie_prob,
        "wins": wins,
        "ties": ties,
        "n_trials": n_trials,
        "win_ci95": ci,
        "category_hint": category_names,
    }