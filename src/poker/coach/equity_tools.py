import random
import copy
from src.poker.core.hand_eval import HandEvaluator
from src.poker.core.deck import Deck
from src.poker.core.cards import Card

def estimate_equity(my_hand, board, villain_range='random', n_samples=5000):
    """
    Monte Carlo equity estimation.
    
    Args:
        my_hand: List of 2 Card objects
        board: List of 0-5 Card objects
        villain_range: 'random' for now (can extend to ranges)
        n_samples: Number of simulations
    
    Returns:
        equity: Float between 0 and 1
    """
    wins = 0
    ties = 0
    
    # Cards already dealt
    known_cards = set(my_hand + board)
    
    for _ in range(n_samples):
        # Create deck without known cards
        deck = Deck()
        deck.cards = [c for c in deck.cards if c not in known_cards]
        deck.shuffle()
        
        # Deal villain hand
        villain_hand = deck.draw(2)
        
        # Complete board
        remaining_board = 5 - len(board)
        simulated_board = board + deck.draw(remaining_board) if remaining_board > 0 else board
        
        # Evaluate
        result = HandEvaluator.compare(my_hand, villain_hand, simulated_board)
        
        if result > 0:
            wins += 1
        elif result == 0:
            ties += 1
    
    equity = (wins + 0.5 * ties) / n_samples
    return equity

def compute_pot_odds(pot, to_call):
    """
    Calculate pot odds.
    
    Args:
        pot: Current pot size
        to_call: Amount to call
    
    Returns:
        pot_odds: Float (e.g., 0.25 means 25% pot odds)
    """
    if to_call == 0:
        return 0
    return to_call / (pot + to_call)

def compute_minimum_defense_frequency(bet_size, pot):
    """
    Calculate MDF (Minimum Defense Frequency) against a bet.
    MDF = Pot / (Pot + Bet)
    
    Args:
        bet_size: Size of the bet
        pot: Pot size before bet
    
    Returns:
        mdf: Float between 0 and 1
    """
    if bet_size == 0:
        return 0
    return pot / (pot + bet_size)
