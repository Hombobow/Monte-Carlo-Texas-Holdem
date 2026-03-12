from typing import List, Tuple
from itertools import combinations
from collections import Counter

Card = Tuple[int, str]

CATEGORY_NAMES = {
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Full House",
    5: "Flush",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "One Pair",
    0: "High Card",
}


def _straight_high(ranks: List[int]) -> int | None:
    """
    Return high card of straight, else None.
    Supports wheel straight A-2-3-4-5 (returns 5).
    """
    # sorted(set(ranks)) is the list of unique ranks in the hand in ascending order
    uniq = sorted(set(ranks))
    
    # if the hand is not 5 cards, return None
    if len(uniq) != 5:
        return None
    
    # if the hand is a wheel straight, return 5
    if uniq == [2, 3, 4, 5, 14]:
        return 5
    
    # if the hand is a straight, return the highest rank
    if max(uniq) - min(uniq) == 4:
        return max(uniq)
    
    return None


def rank_5(cards5: List[Card]) -> Tuple:
    # if the hand is not 5 cards, raise ValueError
    if len(cards5) != 5:
        raise ValueError("rank_5 requires exactly 5 cards.")

    # ranks is the list of ranks in the hand in descending order
    ranks = sorted((r for r, _ in cards5), reverse=True)
    
    # suits is the list of suits in the hand in ascending order
    suits = [s for _, s in cards5]
    
    # counts is the count of each rank in the hand
    # Counter is a dictionary that counts the number of each rank in the hand
    # counts becomes like {14: 2, 13: 2, 2: 1} if the hand is 14, 14, 13, 13, 2
    counts = Counter(ranks)
    
    # is_flush is True if the hand is a flush
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    # Sort by frequency first, then rank desc (for kicker logic)
    # Example: [A,A,K,K,2] -> [(2,A),(2,K),(1,2)]
    by_count_then_rank = sorted(
        # counts.items() gives (rank, frequency) pairs.
        counts.items(), key=lambda x: (x[1], x[0]), reverse=True
    )
    freq_pattern = sorted(counts.values(), reverse=True)

    # Straight flush
    if is_flush and straight_high is not None:
        return (8, straight_high)

    # Four of a kind
    if freq_pattern == [4, 1]:
        quad_rank = by_count_then_rank[0][0]
        kicker = by_count_then_rank[1][0]
        return (7, quad_rank, kicker)

    # Full house
    if freq_pattern == [3, 2]:
        trips = by_count_then_rank[0][0]
        pair = by_count_then_rank[1][0]
        return (6, trips, pair)

    # Flush
    if is_flush:
        # if ranks are [14, 11, 9, 6, 2] result is (5, 14, 11, 9, 6, 2)
        # With *: (5, *[1,2,3]) → (5,1,2,3)
        # Without *: (5, [1,2,3]) → (5, [1,2,3]) (second element is a list, not flattened)
        return (5, *sorted(ranks, reverse=True))

    # Straight
    if straight_high is not None:
        return (4, straight_high)

    # Three of a kind
    if freq_pattern == [3, 1, 1]:
        trips = by_count_then_rank[0][0]
        kickers = sorted([r for r, c in counts.items() if c == 1], reverse=True)
        return (3, trips, *kickers)

    # Two pair
    if freq_pattern == [2, 2, 1]:
        pairs = sorted([r for r, c in counts.items() if c == 2], reverse=True)
        kicker = [r for r, c in counts.items() if c == 1][0]
        return (2, pairs[0], pairs[1], kicker)

    # One pair
    if freq_pattern == [2, 1, 1, 1]:
        pair = [r for r, c in counts.items() if c == 2][0]
        kickers = sorted([r for r, c in counts.items() if c == 1], reverse=True)
        return (1, pair, *kickers)

    # High card
    return (0, *sorted(ranks, reverse=True))


def rank_7(cards7: List[Card]) -> Tuple:
    if len(cards7) != 7:
        raise ValueError("rank_7 requires exactly 7 cards.")
    return max(rank_5(list(c5)) for c5 in combinations(cards7, 5))


def hand_category_name(rank_tuple: Tuple) -> str:
    return CATEGORY_NAMES.get(rank_tuple[0], "Unknown")