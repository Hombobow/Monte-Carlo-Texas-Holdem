from typing import List, Tuple
import random

Card = Tuple[int, str]  # (rank, suit), rank 2..14, suit in "SHDC"
RANK_TO_CHAR = {11: "J", 12: "Q", 13: "K", 14: "A"}


def full_deck() -> List[Card]:
    # return all 52 unique cards
    return [(rank, suit) for rank in range(2, 15) for suit in "SHDC"]


def validate_unique(cards: List[Card]) -> None:
    # raise ValueError if duplicates exist
    if len(cards) != len(set(cards)):
        raise ValueError("Duplicate card detected.")


def remove_known_cards(deck: List[Card], known_cards: List[Card]) -> List[Card]:
    # remove known cards from deck
    known = set(known_cards)
    return [c for c in deck if c not in known]


def draw_without_replacement(deck: List[Card], k: int, rng: random.Random | None = None) -> List[Card]:
    # sample k cards uniformly without replacement
    rng = rng or random
    if k < 0 or k > len(deck):
        raise ValueError("Invalid draw size.")
    return rng.sample(deck, k)


def card_to_str(card: Card) -> str:
    # convert card to string
    rank, suit = card
    rank_char = RANK_TO_CHAR.get(rank, str(rank))
    return f"{rank_char}{suit}"


def cards_to_str(cards: List[Card]) -> str:
    # convert list of cards to string
    return " ".join(card_to_str(c) for c in cards)