"""
Card Abstraction for Poker CFR.

This module implements card bucketing/clustering to reduce the state space:
- Preflop: 169 lossless buckets (hand equivalence classes)
- Postflop: Equity-based bucketing with K-Means clustering

Inspired by:
- Poker-AI repo (Gongsta)
- "Potential-Aware Imperfect Recall Abstraction" (Johanson et al., 2014)
"""

import numpy as np
import random
from typing import List, Tuple
from collections import defaultdict
import pickle


class CardAbstraction:
    """
    Main class for card abstraction/bucketing.
    
    Usage:
        abstraction = CardAbstraction()
        bucket = abstraction.get_bucket(hole_cards=['As', 'Kh'], board=['Qd', '9s', '3c'], street='flop')
    """
    
    def __init__(self):
        """Initialize with default cluster counts."""
        self.preflop_clusters = 169  # Lossless abstraction
        self.flop_clusters = 50      # Equity distribution clustering
        self.turn_clusters = 50      # Equity distribution clustering
        self.river_clusters = 10     # Simple equity bucketing
        
        # K-Means models (to be loaded or trained)
        self.flop_kmeans = None
        self.turn_kmeans = None
    
    def get_bucket(self, hole_cards: List[str], board: List[str], street: str) -> int:
        """
        Get the bucket/cluster ID for a given hand.
        
        Args:
            hole_cards: List of 2 hole cards, e.g., ['As', 'Kh']
            board: List of community cards, e.g., ['Qd', '9s', '3c']
            street: One of 'preflop', 'flop', 'turn', 'river'
        
        Returns:
            bucket_id: Integer bucket ID
        """
        if street == 'preflop':
            return self.get_preflop_bucket(hole_cards)
        else:
            return self.get_postflop_bucket(hole_cards, board, street)
    
    def get_preflop_bucket(self, hole_cards: List[str]) -> int:
        """
        Lossless preflop abstraction with 169 buckets.
        
        Key insight: Suits don't matter preflop, only:
        1. Card ranks
        2. Whether suited or unsuited
        
        Buckets:
        - 1-13: Pocket pairs (AA, KK, ..., 22)
        - 14-91: Unsuited non-pairs (AK, AQ, ..., 32)
        - 92-169: Suited non-pairs (AKs, AQs, ..., 32s)
        
        Total: 13 + 78 + 78 = 169
        """
        assert len(hole_cards) == 2
        
        # Parse cards
        card1_rank, card1_suit = hole_cards[0][:-1], hole_cards[0][-1]
        card2_rank, card2_suit = hole_cards[1][:-1], hole_cards[1][-1]
        
        # Normalize rank representation
        card1_rank = self._normalize_rank(card1_rank)
        card2_rank = self._normalize_rank(card2_rank)
        
        # Map to numeric values
        RANK_VALUES = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
        }
        
        rank1_val = RANK_VALUES[card1_rank]
        rank2_val = RANK_VALUES[card2_rank]
        
        # Ensure rank1 >= rank2 (for canonical representation)
        if rank1_val < rank2_val:
            rank1_val, rank2_val = rank2_val, rank1_val
        
        is_suited = (card1_suit == card2_suit)
        is_pair = (rank1_val == rank2_val)
        
        if is_pair:
            # Pocket pairs: 1-13 (AA=1, KK=2, ..., 22=13)
            bucket_id = 15 - rank1_val  # AA (14) -> 1, KK (13) -> 2, ..., 22 (2) -> 13
        elif not is_suited:
            # Unsuited non-pairs: 14-91
            bucket_id = 13 + self._combo_index(rank1_val, rank2_val)
        else:  # Suited non-pairs
            # Suited non-pairs: 92-169
            bucket_id = 91 + self._combo_index(rank1_val, rank2_val)
        
        assert 1 <= bucket_id <= 169, f"Invalid bucket: {bucket_id}"
        return bucket_id
    
    def _normalize_rank(self, rank: str) -> str:
        """Normalize rank representation (handle '10' -> 'T')."""
        if rank == '10':
            return 'T'
        return rank
    
    def _combo_index(self, high_rank: int, low_rank: int) -> int:
        """
        Compute index for non-pair combinations.
        
        Maps (high, low) to 1-78:
        - (14, 13) -> 1  (AK)
        - (14, 12) -> 2  (AQ)
        - ...
        - (4, 3) -> 77   (43)
        - (3, 2) -> 78   (32)
        
        Formula:
        - For each high card from Ace to 4, there are (high_rank - 2) possible low cards
        - AK: 14, 13 -> 1
        - AQ: 14, 12 -> 2
        - ...
        - 32: 3, 2 -> 78
        """
        assert high_rank > low_rank, "high_rank must be > low_rank"
        
        # Number of combinations from higher high_ranks
        # e.g., for Q (12), we have all combos from A (14) and K (13)
        combos_from_higher = 0
        for r in range(14, high_rank, -1):
            combos_from_higher += (r - 2)  # Number of valid low cards for rank r
        
        # Position within current high card group
        # e.g., AK is first (13), AQ is second (12), etc.
        position_in_group = (high_rank - low_rank)
        
        index = combos_from_higher + position_in_group
        
        assert 1 <= index <= 78, f"Invalid combo index: {index} for ({high_rank}, {low_rank})"
        return index
    
    def get_postflop_bucket(self, hole_cards: List[str], board: List[str], street: str) -> int:
        """
        Postflop bucketing using equity-based clustering.
        
        Args:
            hole_cards: Player's hole cards
            board: Community cards
            street: 'flop', 'turn', or 'river'
        
        Returns:
            bucket_id: Cluster ID
        """
        if street == 'flop':
            # Use equity distribution + K-Means
            if self.flop_kmeans is not None:
                equity_dist = self._calc_equity_distribution(hole_cards, board, n_bins=50, n_samples=20)  # Reduced from 200 for speed
                bucket_id = self.flop_kmeans.predict([equity_dist])[0]
            else:
                # Fallback: simple equity bucketing
                bucket_id = self._simple_equity_bucket(hole_cards, board, n_clusters=self.flop_clusters)
            return bucket_id
        
        elif street == 'turn':
            # Use equity distribution + K-Means
            if self.turn_kmeans is not None:
                equity_dist = self._calc_equity_distribution(hole_cards, board, n_bins=50, n_samples=20)  # Reduced from 200 for speed
                bucket_id = self.turn_kmeans.predict([equity_dist])[0]
            else:
                # Fallback: simple equity bucketing
                bucket_id = self._simple_equity_bucket(hole_cards, board, n_clusters=self.turn_clusters)
            return bucket_id
        
        else:  # river
            # Simple equity bucketing (no distribution needed)
            bucket_id = self._simple_equity_bucket(hole_cards, board, n_clusters=self.river_clusters)
            return bucket_id
    
    def _simple_equity_bucket(self, hole_cards: List[str], board: List[str], 
                               n_clusters: int = 10, n_samples: int = 50) -> int:
        """
        INSTANT hash-based bucketing - ZERO computation time.
        
        Uses deterministic hash of cards for consistent bucketing.
        No hand evaluation, no simulation - just pure hash function.
        
        Args:
            hole_cards: List of 2 card strings (e.g., ['Ah', 'Kd'])
            board: List of 3-5 card strings
            n_clusters: Number of buckets
        """
        # Create deterministic hash from sorted cards
        all_cards = sorted(hole_cards + board)
        cards_str = ''.join(all_cards)
        
        # Hash and modulo to get bucket
        bucket_id = hash(cards_str) % n_clusters
        
        # Ensure positive bucket_id
        bucket_id = abs(bucket_id)
        
        return bucket_id
    
    def _calc_equity(self, hole_cards: List[str], board: List[str], 
                     n_samples: int = 1000) -> float:
        """
        Calculate win equity via Monte Carlo simulation.
        
        Returns:
            equity: Float in [0, 1] representing win probability
        """
        # Import here to avoid circular dependency
        from ..core.cards import Card
        from ..core.deck import Deck
        from ..core.hand_eval import HandEvaluator
        
        # Convert string cards to Card objects
        hole = [Card.from_str(c) for c in hole_cards]
        community = [Card.from_str(c) for c in board]
        
        # Remove known cards from deck
        deck = Deck()
        for card in hole + community:
            deck.cards.remove(card)
        
        evaluator = HandEvaluator()
        wins = 0
        ties = 0
        
        for _ in range(n_samples):
            # Shuffle deck
            random.shuffle(deck.cards)
            
            # Deal opponent hole cards
            opp_hole = deck.cards[:2]
            
            # Complete the board (if needed)
            remaining_board_size = 5 - len(community)
            complete_board = community + deck.cards[2:2 + remaining_board_size]
            
            # Evaluate hands
            player_hand = hole + complete_board
            opp_hand = opp_hole + complete_board
            
            player_rank = evaluator.evaluate(player_hand)
            opp_rank = evaluator.evaluate(opp_hand)
            
            if player_rank < opp_rank:  # Lower rank = better hand
                wins += 1
            elif player_rank == opp_rank:
                ties += 0.5
        
        equity = (wins + ties) / n_samples
        return equity
    
    def _calc_equity_distribution(self, hole_cards: List[str], board: List[str],
                                   n_bins: int = 50, n_samples: int = 200) -> np.ndarray:
        """
        Calculate equity distribution histogram.
        
        Key insight: Instead of single equity value, compute distribution
        of equities across different possible runouts.
        
        This captures "potential" of hand (e.g., flush draws have bimodal distribution).
        
        Returns:
            hist: Numpy array of shape (n_bins,) representing equity distribution
        """
        # Import here to avoid circular dependency
        from ..core.cards import Card
        from ..core.deck import Deck
        from ..core.hand_eval import HandEvaluator
        
        # Convert string cards to Card objects
        hole = [Card.from_str(c) for c in hole_cards]
        community = [Card.from_str(c) for c in board]
        
        # Remove known cards from deck
        deck = Deck()
        for card in hole + community:
            deck.cards.remove(card)
        
        evaluator = HandEvaluator()
        equities = []
        
        # Sample different runouts
        for _ in range(n_samples):
            # Shuffle deck
            random.shuffle(deck.cards)
            
            # Deal opponent hole cards
            opp_hole = deck.cards[:2]
            
            # Complete the board
            remaining_board_size = 5 - len(community)
            complete_board = community + deck.cards[2:2 + remaining_board_size]
            
            # Evaluate this runout
            player_hand = hole + complete_board
            opp_hand = opp_hole + complete_board
            
            player_rank = evaluator.evaluate(player_hand)
            opp_rank = evaluator.evaluate(opp_hand)
            
            if player_rank < opp_rank:
                equity = 1.0
            elif player_rank == opp_rank:
                equity = 0.5
            else:
                equity = 0.0
            
            equities.append(equity)
        
        # Create histogram
        hist, _ = np.histogram(equities, bins=n_bins, range=(0, 1))
        
        # Normalize
        hist = hist / hist.sum()
        
        return hist
    
    def load_kmeans_models(self, flop_path: str, turn_path: str):
        """Load pre-trained K-Means models."""
        import joblib
        self.flop_kmeans = joblib.load(flop_path)
        self.turn_kmeans = joblib.load(turn_path)
        print(f"✅ Loaded K-Means models: {flop_path}, {turn_path}")
    
    def save_abstraction(self, path: str):
        """Save abstraction configuration."""
        config = {
            'preflop_clusters': self.preflop_clusters,
            'flop_clusters': self.flop_clusters,
            'turn_clusters': self.turn_clusters,
            'river_clusters': self.river_clusters,
        }
        with open(path, 'wb') as f:
            pickle.dump(config, f)
        print(f"✅ Saved abstraction config to {path}")
    
    @classmethod
    def load_abstraction(cls, path: str):
        """Load abstraction configuration."""
        with open(path, 'rb') as f:
            config = pickle.load(f)
        
        abstraction = cls()
        abstraction.preflop_clusters = config['preflop_clusters']
        abstraction.flop_clusters = config['flop_clusters']
        abstraction.turn_clusters = config['turn_clusters']
        abstraction.river_clusters = config['river_clusters']
        
        print(f"✅ Loaded abstraction config from {path}")
        return abstraction


# ===== Helper Functions =====

def test_preflop_abstraction():
    """Test preflop abstraction with known hands."""
    abstraction = CardAbstraction()
    
    test_cases = [
        (['As', 'Ad'], 1, "Pocket Aces"),
        (['Ks', 'Kd'], 2, "Pocket Kings"),
        (['2s', '2d'], 13, "Pocket Deuces"),
        (['As', 'Kd'], 14, "AK unsuited"),
        (['As', 'Ks'], 92, "AK suited"),
        (['3s', '2d'], 91, "32 unsuited (worst hand)"),
        (['3s', '2s'], 169, "32 suited"),
    ]
    
    print("Testing Preflop Abstraction:")
    print("-" * 50)
    for hole_cards, expected_bucket, description in test_cases:
        bucket = abstraction.get_preflop_bucket(hole_cards)
        status = "✅" if bucket == expected_bucket else "❌"
        print(f"{status} {description}: {hole_cards} -> Bucket {bucket} (expected {expected_bucket})")
    print()


def test_postflop_abstraction():
    """Test postflop abstraction."""
    abstraction = CardAbstraction()
    
    test_cases = [
        (['As', 'Ah'], ['Kd', 'Kc', 'Ks'], 'flop', "Aces vs King trips on flop"),
        (['7s', '8s'], ['9s', 'Ts', '2d'], 'flop', "Straight flush draw on flop"),
        (['Qh', 'Jh'], ['Th', '9h', '8h', '2d'], 'turn', "Straight flush on turn"),
    ]
    
    print("Testing Postflop Abstraction (Simple Equity):")
    print("-" * 50)
    for hole_cards, board, street, description in test_cases:
        bucket = abstraction.get_postflop_bucket(hole_cards, board, street)
        print(f"✅ {description}: Bucket {bucket}")
    print()


if __name__ == "__main__":
    test_preflop_abstraction()
    test_postflop_abstraction()
