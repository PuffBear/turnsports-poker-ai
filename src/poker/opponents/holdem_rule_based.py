import numpy as np
import random
from src.poker.core.hand_eval import HandEvaluator

class HoldemOpponent:
    def get_action(self, env, obs):
        raise NotImplementedError

class RandomOpponent(HoldemOpponent):
    def get_action(self, env, obs):
        legal = env._get_legal_actions()
        return random.choice(legal)

class TAGOpponent(HoldemOpponent):
    """
    Tight Aggressive:
    - Preflop: Play top 20% hands. Raise if strong, Call if medium.
    - Postflop: C-bet if hit or strong draw. Fold if weak.
    """
    def get_action(self, env, obs):
        legal = env._get_legal_actions()
        
        # Cheat: look at my hand
        my_hand = env.hands[env.current_player]
        board = env.board
        
        # Simple logic
        score = 0
        if len(board) == 0:
            # Preflop
            # Evaluate hole cards strength (approx)
            # High cards good, pairs good
            ranks = [c.rank_index for c in my_hand]
            is_pair = ranks[0] == ranks[1]
            high_card = max(ranks)
            
            if is_pair and high_card >= 8: # 10+ pair
                # Raise
                if env.ACT_RAISE_HALF_POT in legal:
                    return env.ACT_RAISE_HALF_POT
                return env.ACT_CHECK_CALL
            elif high_card >= 10: # J+ high
                return env.ACT_CHECK_CALL
            else:
                if env.ACT_CHECK_CALL in legal:
                    # Check if free
                    if env.street_investment[0] == env.street_investment[1]:
                        return env.ACT_CHECK_CALL
                return env.ACT_FOLD
        else:
            # Postflop
            # Check hand strength
            rank_cat, _ = HandEvaluator.evaluate(my_hand + board)
            
            if rank_cat >= 1: # Pair or better
                # Bet/Raise
                if env.ACT_RAISE_HALF_POT in legal:
                    return env.ACT_RAISE_HALF_POT
                return env.ACT_CHECK_CALL
            else:
                # Check/Fold
                if env.ACT_CHECK_CALL in legal:
                     if env.street_investment[0] == env.street_investment[1]:
                        return env.ACT_CHECK_CALL
                return env.ACT_FOLD
        
        return env.ACT_CHECK_CALL

class LAGOpponent(HoldemOpponent):
    """
    Loose Aggressive:
    - Plays wide range.
    - Bets frequently.
    """
    def get_action(self, env, obs):
        legal = env._get_legal_actions()
        
        # 50% chance to just raise if legal
        if random.random() < 0.5:
            raises = [a for a in legal if a >= 2]
            if raises:
                return random.choice(raises)
        
        return env.ACT_CHECK_CALL
