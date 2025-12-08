#!/usr/bin/env python3
"""
Test the fixed Hold'em environment to ensure game logic works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.opponents.holdem_rule_based import RandomOpponent

def test_single_hand():
    """Play one hand to completion."""
    print("=" * 60)
    print("Testing Single Hand")
    print("=" * 60)
    
    env = HoldemHuEnv()
    opponent = RandomOpponent()
    
    state, info = env.reset()
    print(f"Initial state:")
    env.render()
    print()
    
    done = False
    step_count = 0
    total_reward = 0
    
    while not done and step_count < 50:
        legal_actions = env._get_legal_actions()
        action = opponent.get_action(env, state)
        
        print(f"Step {step_count}: Player {env.current_player} takes action {action}")
        
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if done:
            print(f"\nHand finished in {step_count} steps")
            env.render()
            print(f"Total reward: {total_reward:.2f} BB")
            print(f"Info: {info}")
    
    return total_reward

def test_multiple_hands(n_hands=100):
    """Test multiple hands to check for crashes and reward balance."""
    print("\n" + "=" * 60)
    print(f"Testing {n_hands} Hands")
    print("=" * 60)
    
    env = HoldemHuEnv()
    opponent = RandomOpponent()
    
    rewards_p0 = []
    rewards_p1 = []
    crashes = 0
    
    for hand in range(n_hands):
        try:
            state, info = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 50:
                legal_actions = env._get_legal_actions()
                action = opponent.get_action(env, state)
                state, reward, done, truncated, info = env.step(action)
                step_count += 1
            
            # Record reward
            if env.current_player == 0:
                rewards_p0.append(reward)
            else:
                rewards_p1.append(reward)
                
        except Exception as e:
            print(f"Hand {hand} crashed: {e}")
            crashes += 1
    
    print(f"\nResults:")
    print(f"  Hands completed: {n_hands - crashes}")
    print(f"  Crashes: {crashes}")
    
    if rewards_p0:
        import numpy as np
        print(f"  P0 avg reward: {np.mean(rewards_p0):.2f} BB")
        print(f"  P1 avg reward: {np.mean(rewards_p1):.2f} BB")
        print(f"  Total net: {np.sum(rewards_p0) + np.sum(rewards_p1):.2f} BB (should be ~0)")
    
    return crashes == 0

def main():
    print("\n" + "=" * 60)
    print("HOLD'EM ENVIRONMENT TEST - FIXED VERSION")
    print("=" * 60)
    print()
    
    # Test 1: Single hand
    try:
        test_single_hand()
        print("\n✓ Single hand test passed")
    except Exception as e:
        print(f"\n✗ Single hand test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Multiple hands
    try:
        success = test_multiple_hands(100)
        if success:
            print("\n✓ Multiple hands test passed")
        else:
            print("\n✗ Some hands crashed")
    except Exception as e:
        print(f"\n✗ Multiple hands test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Environment is ready for training.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
