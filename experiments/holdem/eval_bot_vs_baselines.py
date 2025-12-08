import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from tqdm import tqdm
from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.agents.holdem_dqn import DQNAgent
from src.poker.opponents.holdem_rule_based import RandomOpponent, TAGOpponent, LAGOpponent

def evaluate_agent(agent, opponent, n_hands=1000, verbose=True):
    """
    Evaluate agent against an opponent.
    
    Returns:
        dict with metrics: win_rate, avg_reward, bb_per_100
    """
    env = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
    
    total_reward = 0
    wins = 0
    losses = 0
    ties = 0
    
    for hand in tqdm(range(n_hands), disable=not verbose, desc=f"vs {opponent.__class__.__name__}"):
        state, info = env.reset()
        agent_player = 0
        done = False
        
        while not done:
            current_player = env.current_player
            legal_actions = env._get_legal_actions()
            
            if current_player == agent_player:
                # Agent plays deterministically
                action = agent.select_action(state, legal_actions, epsilon=0.0)
            else:
                # Opponent plays
                action = opponent.get_action(env, state)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            if current_player == agent_player and done:
                total_reward += reward
                
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    ties += 1
            
            state = next_state
    
    # Calculate metrics
    win_rate = wins / n_hands
    bb_per_100 = (total_reward / n_hands) * 100
    avg_reward = total_reward / n_hands
    
    return {
        'opponent': opponent.__class__.__name__,
        'n_hands': n_hands,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'bb_per_100': bb_per_100
    }

def main():
    """Evaluate trained agent against all baseline opponents."""
    
    print("=" * 70)
    print("Evaluating DQN Agent vs Baseline Opponents")
    print("=" * 70)
    
    # Load agent
    agent_path = 'checkpoints/dqn_final.pt'
    
    if not os.path.exists(agent_path):
        print(f"\nError: No trained agent found at {agent_path}")
        print("Train an agent first with: python experiments/holdem/train_holdem_dqn_vs_pool.py")
        return
    
    print(f"\nLoading agent from {agent_path}...")
    agent = DQNAgent(state_dim=200, action_dim=9, device='cpu')
    agent.load(agent_path)
    print("✓ Agent loaded successfully")
    
    # Opponents
    opponents = [
        RandomOpponent(),
        TAGOpponent(),
        LAGOpponent()
    ]
    
    # Evaluation
    n_hands = 10000  # 10k hands per opponent for statistical significance
    results = []
    
    print(f"\nEvaluating over {n_hands} hands per opponent...\n")
    
    for opponent in opponents:
        result = evaluate_agent(agent, opponent, n_hands=n_hands, verbose=True)
        results.append(result)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['opponent']}:")
        print(f"  Hands:      {result['n_hands']}")
        print(f"  Win Rate:   {result['win_rate']*100:.2f}%")
        print(f"  Wins:       {result['wins']}")
        print(f"  Losses:     {result['losses']}")
        print(f"  Ties:       {result['ties']}")
        print(f"  Avg Reward: {result['avg_reward']:+.3f} BB/hand")
        print(f"  bb/100:     {result['bb_per_100']:+.2f}")
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    total_hands = sum(r['n_hands'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    total_reward = sum(r['avg_reward'] * r['n_hands'] for r in results)
    
    overall_win_rate = total_wins / total_hands
    overall_bb_100 = (total_reward / total_hands) * 100
    
    print(f"Total Hands:        {total_hands}")
    print(f"Overall Win Rate:   {overall_win_rate*100:.2f}%")
    print(f"Overall bb/100:     {overall_bb_100:+.2f}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if overall_bb_100 > 5:
        print("✓ Excellent: Agent is crushing the opponents!")
    elif overall_bb_100 > 2:
        print("✓ Good: Agent is profitable")
    elif overall_bb_100 > 0:
        print("○ Okay: Agent is slightly profitable")
    elif overall_bb_100 > -2:
        print("⚠ Fair: Agent is near break-even")
    else:
        print("✗ Poor: Agent needs more training")
    
    print("\nNote: Professional players aim for 5-10 bb/100 in live games.")
    print("      Online winners typically achieve 3-7 bb/100.")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
