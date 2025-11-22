"""
Action Abstraction for Poker CFR.

Simplifies bet sizing to discrete actions, reducing game tree complexity.

Key insight from Poker-AI repo:
- Limit to 11 betting sequences per street
- Actions: fold, check, call, bet_small (1/3 pot), bet_large (pot-size)

This prevents infinite raise sequences and makes CFR tractable.
"""

from typing import List, Tuple
from enum import Enum


class Action(Enum):
    """Discrete poker actions."""
    FOLD = 'f'
    CHECK = 'k'  # 'k' for 'check' (Poker-AI convention)
    CALL = 'c'
    BET_MIN = 'bMIN'  # 1/3 pot or min raise
    BET_MAX = 'bMAX'  # Pot-size bet


class ActionAbstraction:
    """
    Simplifies continuous bet sizing to discrete actions.
    
    Maps 9-action Hold'em environment to 5-action abstracted game:
    - fold, check, call, bet_small, bet_large
    
    Limits betting sequences to prevent infinite game trees.
    """
    
    # Discrete actions for CFR
    DISCRETE_ACTIONS = [Action.FOLD, Action.CHECK, Action.CALL, Action.BET_MIN, Action.BET_MAX]
    
    # Valid betting sequences (from Poker-AI)
    # This limits the game tree to 11 sequences per street
    VALID_SEQUENCES = [
        ['k', 'k'],          # Check-check
        ['k', 'bMIN', 'f'],  # Check, small bet, fold
        ['k', 'bMIN', 'c'],  # Check, small bet, call
        ['k', 'bMAX', 'f'],  # Check, large bet, fold
        ['k', 'bMAX', 'c'],  # Check, large bet, call
        ['bMIN', 'f'],       # Small bet, fold
        ['bMIN', 'c'],       # Small bet, call
        ['bMIN', 'bMAX', 'f'],  # Small bet, raise large, fold
        ['bMIN', 'bMAX', 'c'],  # Small bet, raise large, call
        ['bMAX', 'f'],       # Large bet, fold
        ['bMAX', 'c'],       # Large bet, call
    ]
    
    def __init__(self, min_bet_frac: float = 1/3, max_bet_frac: float = 1.0):
        """
        Initialize action abstraction.
        
        Args:
            min_bet_frac: Fraction of pot for small bets (default 1/3)
            max_bet_frac: Fraction of pot for large bets (default 1.0 = pot-size)
        """
        self.min_bet_frac = min_bet_frac
        self.max_bet_frac = max_bet_frac
    
    def abstract_action(self, raw_action_idx: int, pot: float, to_call: float) -> Action:
        """
        Map raw 9-action environment action to discrete abstracted action.
        
        9-action mapping (from holdem_hu_env.py):
        0: fold
        1: check/call
        2: min-raise
        3: raise 0.25 pot
        4: raise 0.33 pot
        5: raise 0.5 pot
        6: raise 0.75 pot
        7: raise 1.0 pot (pot-size)
        8: all-in
        
        Args:
            raw_action_idx: Action index from environment (0-8)
            pot: Current pot size
            to_call: Amount to call
        
        Returns:
            Abstract action
        """
        if raw_action_idx == 0:
            return Action.FOLD
        
        elif raw_action_idx == 1:
            # Check or call
            if to_call == 0:
                return Action.CHECK
            else:
                return Action.CALL
        
        elif raw_action_idx in [2, 3, 4]:
            # Small raises (min, 0.25, 0.33 pot) -> BET_MIN
            return Action.BET_MIN
        
        elif raw_action_idx in [5, 6, 7, 8]:
            # Large raises (0.5, 0.75, pot, all-in) -> BET_MAX
            return Action.BET_MAX
        
        else:
            raise ValueError(f"Invalid action index: {raw_action_idx}")
    
    def expand_action(self, abstract_action: Action, pot: float, stack: float, 
                      to_call: float, min_raise: float) -> int:
        """
        Map abstract action back to concrete environment action.
        
        Args:
            abstract_action: Abstracted action
            pot: Current pot size
            stack: Player's stack size
            to_call: Amount needed to call
            min_raise: Minimum raise amount
        
        Returns:
            Environment action index (0-8)
        """
        if abstract_action == Action.FOLD:
            return 0
        
        elif abstract_action == Action.CHECK:
            return 1  # Check (if to_call == 0)
        
        elif abstract_action == Action.CALL:
            return 1  # Call
        
        elif abstract_action == Action.BET_MIN:
            # Map to 1/3 pot raise (action 4)
            # But if pot is small, use min-raise (action 2)
            min_bet_size = max(min_raise, pot * self.min_bet_frac)
            
            if min_bet_size < pot * 0.25:
                return 2  # Min-raise
            else:
                return 4  # 1/3 pot raise
        
        elif abstract_action == Action.BET_MAX:
            # Map to pot-size raise (action 7)
            # But check stack size
            pot_size_bet = pot * self.max_bet_frac
            
            if pot_size_bet >= stack:
                return 8  # All-in
            else:
                return 7  # Pot-size raise
        
        else:
            raise ValueError(f"Invalid abstract action: {abstract_action}")
    
    def get_legal_actions(self, history_str: str, to_call: float) -> List[Action]:
        """
        Get legal actions based on betting history.
        
        This implements the 11-sequence limitation from Poker-AI.
        
        Args:
            history_str: Betting history string (e.g., 'k', 'kbMIN', 'bMINc')
            to_call: Amount needed to call
        
        Returns:
            List of legal abstract actions
        """
        # Parse last action
        if len(history_str) == 0:
            # First action: can check or bet
            return [Action.CHECK, Action.BET_MIN, Action.BET_MAX]
        
        last_action = history_str[-1]
        
        if last_action == 'k':
            # After check: can check, bet small, or bet large
            return [Action.CHECK, Action.BET_MIN, Action.BET_MAX]
        
        elif last_action == 'bMIN':
            # After small bet: can fold, call, or raise large
            if to_call > 0:
                return [Action.FOLD, Action.CALL, Action.BET_MAX]
            else:
                return [Action.FOLD, Action.CALL]
        
        elif last_action == 'bMAX':
            # After large bet: can only fold or call (no re-raise)
            return [Action.FOLD, Action.CALL]
        
        elif last_action == 'c':
            # After call: street is over (no more actions)
            return []
        
        elif last_action == 'f':
            # After fold: hand is over
            return []
        
        else:
            # Default: all actions available
            return [Action.CHECK, Action.CALL, Action.BET_MIN, Action.BET_MAX, Action.FOLD]
    
    def is_sequence_valid(self, sequence: List[str]) -> bool:
        """
        Check if a betting sequence is one of the 11 valid sequences.
        
        Args:
            sequence: List of action codes (e.g., ['k', 'bMIN', 'c'])
        
        Returns:
            True if sequence is valid, False otherwise
        """
        return sequence in self.VALID_SEQUENCES
    
    def get_available_actions_env(self, env_legal_actions: List[int], 
                                   pot: float, to_call: float) -> List[Action]:
        """
        Convert environment legal actions to abstract actions.
        
        Args:
            env_legal_actions: List of legal action indices from environment
            pot: Current pot size
            to_call: Amount to call
        
        Returns:
            List of legal abstract actions
        """
        abstract_actions = set()
        
        for action_idx in env_legal_actions:
            abstract_action = self.abstract_action(action_idx, pot, to_call)
            abstract_actions.add(abstract_action)
        
        return list(abstract_actions)
    
    def action_to_string(self, action: Action) -> str:
        """Convert action enum to string code."""
        return action.value
    
    def string_to_action(self, action_str: str) -> Action:
        """Convert string code to action enum."""
        for action in self.DISCRETE_ACTIONS:
            if action.value == action_str:
                return action
        raise ValueError(f"Invalid action string: {action_str}")


# ===== Helper Functions =====

def test_action_abstraction():
    """Test action abstraction mapping."""
    abstraction = ActionAbstraction()
    
    print("Testing Action Abstraction:")
    print("-" * 60)
    
    # Test raw action mapping
    print("\n1. Raw Action -> Abstract Action:")
    test_cases = [
        (0, 100, 0, Action.FOLD, "Fold"),
        (1, 100, 0, Action.CHECK, "Check (no bet)"),
        (1, 100, 50, Action.CALL, "Call (facing bet)"),
        (3, 100, 0, Action.BET_MIN, "0.25 pot -> BET_MIN"),
        (4, 100, 0, Action.BET_MIN, "0.33 pot -> BET_MIN"),
        (7, 100, 0, Action.BET_MAX, "Pot-size -> BET_MAX"),
        (8, 100, 0, Action.BET_MAX, "All-in -> BET_MAX"),
    ]
    
    for action_idx, pot, to_call, expected_action, description in test_cases:
        result = abstraction.abstract_action(action_idx, pot, to_call)
        status = "✅" if result == expected_action else "❌"
        print(f"{status} {description}: Action {action_idx} -> {result.value}")
    
    # Test legal actions
    print("\n2. Legal Actions by History:")
    history_cases = [
        ('', 0, "Start of street"),
        ('k', 0, "After check"),
        ('bMIN', 50, "After small bet"),
        ('bMAX', 100, "After large bet"),
        ('kbMINc', 0, "After check-bet-call"),
    ]
    
    for history, to_call, description in history_cases:
        legal = abstraction.get_legal_actions(history, to_call)
        legal_str = [a.value for a in legal]
        print(f"  {description} ('{history}'): {legal_str}")
    
    # Test valid sequences
    print("\n3. Valid Betting Sequences:")
    for i, seq in enumerate(ActionAbstraction.VALID_SEQUENCES, 1):
        print(f"  {i}. {' -> '.join(seq)}")
    
    print()


if __name__ == "__main__":
    test_action_abstraction()
