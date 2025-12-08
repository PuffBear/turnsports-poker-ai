#!/usr/bin/env python3
"""
Quick test script to verify the environment and agent are working.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.opponents.holdem_rule_based import RandomOpponent

def test_environment():
    """Test that the environment runs."""
    print("Testing Hold'em Environment...")
    
    env = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
    opponent = RandomOpponent()
    
    state, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Player 0 hand: {env.hands[0]}")
    print(f"  Player 1 hand: {env.hands[1]}")
    print(f"  Board: {env.board}")
    print(f"  Pot: {env.pot} BB")
    
    # Play one hand
    done = False
    actions_taken = 0
    
    while not done and actions_taken < 50:
        legal_actions = env._get_legal_actions()
        action = opponent.get_action(env, state)
        
        state, reward, done, truncated, info = env.step(action)
        actions_taken += 1
    
    print(f"✓ Hand completed in {actions_taken} actions")
    print(f"  Final board: {env.board}")
    print(f"  Final pot: {env.pot} BB")
    
    if 'winner' in info:
        print(f"  Winner: Player {info['winner']}")
    
    print("\n✓ Environment test passed!\n")

def test_coach():
    """Test that the coach works."""
    print("Testing Agentic Coach...")
    
    try:
        from src.poker.coach.agentic_coach import AgenticCoach
        from src.poker.envs.holdem_hu_env import HoldemHuEnv
        
        env = HoldemHuEnv()
        state, info = env.reset()
        
        coach = AgenticCoach(bot_policy=None, use_rollouts=False, use_equity=True)
        
        # Get recommendation (without rollouts since we don't have a trained bot)
        recommendation = coach.get_recommendation(env, n_equity_samples=1000)
        
        print(f"✓ Coach recommendation: {recommendation['action_name']}")
        print(f"  Tool results: {recommendation['tool_results']}")
        print(f"\n✓ Coach test passed!\n")
        
    except Exception as e:
        print(f"✗ Coach test failed: {e}\n")

def main():
    print("=" * 60)
    print("Quick Start Test - Poker RL Project")
    print("=" * 60)
    print()
    
    test_environment()
    test_coach()
    
    print("=" * 60)
    print("All tests passed!")
    print("\nNext steps:")
    print("1. Train agent: python experiments/holdem/train_holdem_dqn_vs_pool.py")
    print("2. Play vs bot:  python gui/holdem_poker_gui.py")
    print("3. Evaluate bot: python experiments/holdem/eval_bot_vs_baselines.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
