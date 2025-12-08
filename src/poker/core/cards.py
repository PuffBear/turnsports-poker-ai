class Card:
    SUITS = 'cdhs'
    RANKS = '23456789TJQKA'
    
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.rank_index = self.RANKS.index(rank)
        self.suit_index = self.SUITS.index(suit)

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def to_int(self):
        # Unique integer representation if needed
        return self.suit_index * 13 + self.rank_index

    @staticmethod
    def from_str(card_str):
        return Card(card_str[0], card_str[1])
