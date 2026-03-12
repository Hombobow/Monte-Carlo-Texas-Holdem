import random
from deck import cards_to_str, full_deck, draw_without_replacement
from simulator import simulate_state, simulate_state_unknown_opponents
from visualize import plot_stage_snapshot, plot_challenge_story


def print_state(stage_name, board, labels, result):
    print(f"\n=== {stage_name.upper()} ===")
    print(f"Board: {cards_to_str(board) if board else '(none)'}")
    for i, label in enumerate(labels):
        p = result["win_prob"][i]
        t = result["tie_prob"][i]
        lo, hi = result["win_ci95"][i]
        print(f"{label}: P(win)={p:.4f}, P(tie)={t:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")


def deal_random_hand(n_players, rng):
    if n_players < 2 or n_players > 10:
        raise ValueError("Number of players must be between 2 and 10.")

    cards_needed = (2 * n_players) + 5
    drawn = draw_without_replacement(full_deck(), cards_needed, rng)

    hole_cards = [drawn[2 * i : 2 * i + 2] for i in range(n_players)]
    board_full = drawn[2 * n_players :]
    labels = [f"Player {i + 1}" for i in range(n_players)]
    return hole_cards, board_full, labels


def print_hole_cards(hole_cards, labels):
    print("\n=== HOLE CARDS ===")
    for i, label in enumerate(labels):
        print(f"{label}: {cards_to_str(hole_cards[i])}")


def prompt_folds(active_indices, labels, prompt_label):
    if len(active_indices) <= 1:
        return []
    raw = input(
        f"Players folding {prompt_label} (numbers like 2,4; Enter for none): "
    ).strip()
    if not raw:
        return []

    chosen = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            player_num = int(token)
        except ValueError:
            continue
        idx = player_num - 1
        if idx in active_indices:
            chosen.add(idx)

    if len(chosen) >= len(active_indices):
        print("Cannot fold all remaining players. Ignoring fold input.")
        return []
    return sorted(chosen)


def active_view(hole_cards, labels, active_indices):
    active_hole_cards = [hole_cards[i] for i in active_indices]
    active_labels = [labels[i] for i in active_indices]
    return active_hole_cards, active_labels


def deal_hero_and_board(rng):
    drawn = draw_without_replacement(full_deck(), 7, rng)
    hero_hole = drawn[:2]
    board_full = drawn[2:]
    return hero_hole, board_full


def build_table_context_text(board, labels, hands_by_label):
    board_text = cards_to_str(board) if board else "(none)"
    if not hands_by_label:
        hands_text = "(hidden)"
    else:
        pieces = []
        for label in labels:
            hand = hands_by_label.get(label)
            if hand is None:
                pieces.append(f"{label}: (unknown)")
            else:
                pieces.append(f"{label}: {cards_to_str(hand)}")
        hands_text = " | ".join(pieces)
    return f"Board: {board_text}\nHands: {hands_text}"


def main():
    print("Texas Hold'em Monte Carlo Challenge")

    raw_players = input("How many players? (2-10, default 2): ").strip()
    try:
        n_players = int(raw_players) if raw_players else 2
    except ValueError:
        print("Invalid player count. Using default: 2 players.")
        n_players = 2

    raw_seed = input("Random seed (optional, Enter for random): ").strip()
    try:
        seed = int(raw_seed) if raw_seed else None
    except ValueError:
        print("Invalid seed. Using random seed.")
        seed = None
    rng = random.Random(seed)

    mode_raw = input(
        "Opponent card mode: [1] fixed dealt hands, [2] random unknown each trial (default 2): "
    ).strip()
    unknown_mode = mode_raw != "1"

    hole_cards = None
    hero_hole = None
    labels = []
    show_hands = False
    enable_folding = False
    if unknown_mode:
        hero_hole, board_full = deal_hero_and_board(rng)
        labels = ["Hero"] + [f"Opponent {i}" for i in range(1, n_players)]
        print("\n=== HOLE CARDS ===")
        print(f"Hero: {cards_to_str(hero_hole)}")
        print("Opponents: (unknown)")
    else:
        try:
            hole_cards, board_full, labels = deal_random_hand(n_players, rng)
        except ValueError:
            print("Player count out of range. Using default: 2 players.")
            hole_cards, board_full, labels = deal_random_hand(2, rng)

        show_hands_raw = input("Show all players' hole cards? (y/N): ").strip().lower()
        show_hands = show_hands_raw in {"y", "yes"}
        if show_hands:
            print_hole_cards(hole_cards, labels)
            fold_raw = input(
                "Enable folding prompts (before flop/turn/river and after river)? (y/N): "
            ).strip().lower()
            enable_folding = fold_raw in {"y", "yes"}
        else:
            print("\n=== HOLE CARDS ===")
            print("(hidden)")

    history = {}
    stage_records = []
    trials_by_stage = {
        "preflop": 20000,
        "flop": 20000,
        "turn": 20000,
        "river": 5000,
    }
    streets = [("preflop", 0), ("flop", 3), ("turn", 4), ("river", 5)]
    base_labels = list(labels)
    active_indices = list(range(len(base_labels)))
    last_stage_name = None

    for stage_name, n_board_cards in streets:
        board = board_full[:n_board_cards]
        stage_seed = rng.randint(0, 10**9)
        last_stage_name = stage_name
        if unknown_mode:
            result = simulate_state_unknown_opponents(
                hero_hole,
                n_players,
                board,
                n_trials=trials_by_stage[stage_name],
                seed=stage_seed,
            )
            history[stage_name] = result["win_prob"]
            print_state(stage_name, board, labels, result)
            hands_by_label = {labels[0]: hero_hole}
            for opp_label in labels[1:]:
                hands_by_label[opp_label] = None
            context_text = build_table_context_text(board, labels, hands_by_label)
            stage_records.append(
                {
                    "stage_name": stage_name,
                    "win_prob": list(result["win_prob"]),
                    "tie_prob": list(result["tie_prob"]),
                    "win_ci95": list(result["win_ci95"]),
                    "board_text": cards_to_str(board) if board else "(none)",
                }
            )
            plot_stage_snapshot(stage_name, labels, result, context_text=context_text)
        else:
            if show_hands:
                active_hole_cards, active_labels = active_view(
                    hole_cards, base_labels, active_indices
                )
                print_hole_cards(active_hole_cards, active_labels)

            if enable_folding and stage_name != "preflop":
                folded = prompt_folds(active_indices, base_labels, f"before {stage_name}")
                active_indices = [i for i in active_indices if i not in folded]
                if folded:
                    folded_labels = ", ".join(base_labels[i] for i in folded)
                    print(f"Folded: {folded_labels}")

            if len(active_indices) == 1:
                winner_idx = active_indices[0]
                win_vector = [0.0] * len(base_labels)
                win_vector[winner_idx] = 1.0
                history[stage_name] = win_vector
                tie_vector = [0.0] * len(base_labels)
                ci_vector = [(p, p) for p in win_vector]
                stage_records.append(
                    {
                        "stage_name": stage_name,
                        "win_prob": win_vector,
                        "tie_prob": tie_vector,
                        "win_ci95": ci_vector,
                        "board_text": cards_to_str(board) if board else "(none)",
                    }
                )
                print(f"\n=== {stage_name.upper()} ===")
                print(f"Board: {cards_to_str(board) if board else '(none)'}")
                print(f"{base_labels[winner_idx]} wins by everyone else folding.")
                fold_result = {
                    "win_prob": win_vector,
                    "tie_prob": tie_vector,
                    "win_ci95": ci_vector,
                }
                if show_hands:
                    hands_by_label = {base_labels[i]: hole_cards[i] for i in active_indices}
                else:
                    hands_by_label = {}
                context_text = build_table_context_text(board, base_labels, hands_by_label)
                plot_stage_snapshot(
                    stage_name,
                    base_labels,
                    fold_result,
                    context_text=context_text,
                )
                break

            active_hole_cards, active_labels = active_view(
                hole_cards, base_labels, active_indices
            )
            result = simulate_state(
                active_hole_cards,
                board,
                n_trials=trials_by_stage[stage_name],
                seed=stage_seed,
            )

            print_state(stage_name, board, active_labels, result)
            win_vector = [0.0] * len(base_labels)
            for pos, idx in enumerate(active_indices):
                win_vector[idx] = result["win_prob"][pos]
            history[stage_name] = win_vector
            tie_vector = [0.0] * len(base_labels)
            ci_vector = [(0.0, 0.0)] * len(base_labels)
            for pos, idx in enumerate(active_indices):
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
            if show_hands:
                hands_by_label = {base_labels[i]: hole_cards[i] for i in active_indices}
            else:
                hands_by_label = {}
            context_text = build_table_context_text(board, base_labels, hands_by_label)
            plot_stage_snapshot(
                stage_name,
                base_labels,
                stage_record,
                context_text=context_text,
            )

            if enable_folding and stage_name == "river":
                folded = prompt_folds(active_indices, base_labels, "after river")
                active_indices = [i for i in active_indices if i not in folded]
                if folded:
                    folded_labels = ", ".join(base_labels[i] for i in folded)
                    remaining_labels = ", ".join(base_labels[i] for i in active_indices)
                    print(f"Folded: {folded_labels}")
                    print(f"Remaining players: {remaining_labels}")

    if last_stage_name is not None and stage_records:
        plot_challenge_story(stage_records, base_labels)


if __name__ == "__main__":
    main()