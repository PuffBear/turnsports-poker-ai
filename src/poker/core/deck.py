import random
from src.poker.core.cards import Card

class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for s in Card.SUITS for r in Card.RANKS]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, n=1):
        if n > len(self.cards):
            raise ValueError("Not enough cards in deck")
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def reset(self):
        self.__init__()
