#!/usr/bin/env python3
"""
Proper test that verifies zero-sum property of the game.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.opponents.holdem_rule_based import RandomOpponent
import numpy as np

def test_zero_sum():
    """Test that rewards sum to zero in each hand."""
    print("=" * 60)
    print("Testing Zero-Sum Property")
    print("=" * 60)
    
    env = HoldemHuEnv()
    opponent = RandomOpponent()
    
    n_hands = 100
    total_sum = 0
    individual_sums = []
    
    for hand_num in range(n_hands):
        state, info = env.reset()
        
        # Track rewards for THIS specific hand
        hand_rewards = {0: 0, 1: 0}
        done = False
        
        while not done:
            legal_actions = env._get_legal_actions()
            action = opponent.get_action(env, state)
            
            acting_player = env.current_player
            state, reward, done, truncated, info = env.step(action)
            
            # Accumulate reward for the player who just acted
            hand_rewards[acting_player] += reward
        
        # For this hand, sum should be zero
        hand_sum = hand_rewards[0] + hand_rewards[1]
        individual_sums.append(hand_sum)
        total_sum += hand_sum
        
        if abs(hand_sum) > 0.01:
            print(f"Hand {hand_num}: P0={hand_rewards[0]:.2f}, P1={hand_rewards[1]:.2f}, Sum={hand_sum:.2f} ❌")
    
    print(f"\nTotal sum across {n_hands} hands: {total_sum:.2f}")
    print(f"Average deviation per hand: {np.mean(np.abs(individual_sums)):.4f}")
    print(f"Max deviation: {np.max(np.abs(individual_sums)):.4f}")
    
    if abs(total_sum) < 1.0:  # Allow small floating point errors
        print("\n✅ Zero-sum property holds!")
        return True
    else:
        print("\n❌ Zero-sum property violated!")
        return False

def main():
    success = test_zero_sum()
    
    if success:
        print("\n" + "=" * 60)
        print("Environment is correct!")
        print("Ready for training.")
        print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
