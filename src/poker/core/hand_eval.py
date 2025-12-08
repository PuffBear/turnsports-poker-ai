from collections import Counter
from src.poker.core.cards import Card

class HandEvaluator:
    RANK_VALUES = {r: i for i, r in enumerate(Card.RANKS)}
    
    @staticmethod
    def evaluate(cards):
        # cards: list of Card objects (5 to 7 cards)
        # Returns a tuple (rank_category, tie_breakers)
        # rank_category: 8 (Straight Flush) to 0 (High Card)
        # tie_breakers: list of rank indices to break ties
        
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate")
            
        # We need to find the best 5-card hand
        import itertools
        best_score = (-1, [])
        
        for hand in itertools.combinations(cards, 5):
            score = HandEvaluator._score_5_cards(hand)
            if score > best_score:
                best_score = score
                
        return best_score

    @staticmethod
    def _score_5_cards(hand):
        # hand: 5 Card objects
        ranks = sorted([HandEvaluator.RANK_VALUES[c.rank] for c in hand], reverse=True)
        suits = [c.suit for c in hand]
        is_flush = len(set(suits)) == 1
        
        # Check straight
        is_straight = False
        if len(set(ranks)) == 5:
            if ranks[0] - ranks[4] == 4:
                is_straight = True
            elif ranks == [12, 3, 2, 1, 0]: # A, 5, 4, 3, 2
                is_straight = True
                ranks = [3, 2, 1, 0, -1] # Adjust for wheel
        
        if is_straight and is_flush:
            return (8, ranks)
        
        rank_counts = Counter(ranks)
        counts = rank_counts.most_common()
        # counts is list of (rank, count), sorted by count desc, then insertion order
        # We need to sort by count desc, then rank desc
        counts.sort(key=lambda x: (x[1], x[0]), reverse=True)
        
        if counts[0][1] == 4:
            return (7, [counts[0][0], counts[1][0]])
        
        if counts[0][1] == 3 and counts[1][1] == 2:
            return (6, [counts[0][0], counts[1][0]])
        
        if is_flush:
            return (5, ranks)
        
        if is_straight:
            return (4, ranks)
        
        if counts[0][1] == 3:
            return (3, [counts[0][0]] + [x[0] for x in counts[1:]])
        
        if counts[0][1] == 2 and counts[1][1] == 2:
            return (2, [counts[0][0], counts[1][0], counts[2][0]])
        
        if counts[0][1] == 2:
            return (1, [counts[0][0]] + [x[0] for x in counts[1:]])
        
        return (0, ranks)

    @staticmethod
    def compare(hand1, hand2, board):
        # Returns 1 if hand1 wins, -1 if hand2 wins, 0 if tie
        score1 = HandEvaluator.evaluate(hand1 + board)
        score2 = HandEvaluator.evaluate(hand2 + board)
        
        if score1 > score2:
            return 1
        elif score2 > score1:
            return -1
        else:
            return 0
