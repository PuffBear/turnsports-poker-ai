"""
Training script for RL-CFR on Kuhn Poker.

This demonstrates the RL-CFR concept on a simple game before
scaling to Texas Hold'em.

Kuhn poker:
- 3 cards (J, Q, K)
- 2 players
- 1 betting round
- 2 actions (pass/bet)
- Perfect testbed for CFR algorithms

Expected results:
- After 10,000 iterations: Learns basic strategy
- After 100,000 iterations: Near-optimal strategy
- Exploitability should decrease monotonically
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.poker.agents.kuhn_rlcfr import RLCFRAgent
from src.poker.envs.kuhn_poker_env import KuhnPokerEnv


def train_kuhn_rlcfr(n_iterations: int = 100000, 
                      save_interval: int = 10000,
                      eval_interval: int = 1000,
                      checkpoint_dir: str = 'checkpoints/kuhn_rlcfr'):
    """
    Train RL-CFR agent on Kuhn Poker.
    
    Args:
        n_iterations: Number of training iterations
        save_interval: Save checkpoint every N iterations
        eval_interval: Evaluate every N iterations
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        agent: Trained RL-CFR agent
    """
    print("=" * 70)
    print("Training RL-CFR Agent on Kuhn Poker")
    print("=" * 70)
    print(f"\nIterations: {n_iterations:,}")
    print(f"Save interval: {save_interval:,}")
    print(f"Eval interval: {eval_interval:,}")
    print()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = KuhnPokerEnv()
    
    # Kuhn poker state: [my_card_one_hot(3), pot(1), current_bet(1), history_encoding(1)] = 6 dims
    # Actions: 0=pass/check, 1=bet
    agent = RLCFRAgent(state_dim=6, num_actions=2, hidden_dim=64, learning_rate=1e-3)
    
    # Tracking metrics
    utilities = {'p0': [], 'p1': []}
    exploitabilities = []
    iteration_points = []
    
    # Training loop
    print("Training...")
    print("-" * 70)
    
    for iteration in tqdm(range(n_iterations)):
        # Alternate between players
        # This ensures both players are updated equally
        
        # Player 0 iteration
        env.reset()
        utility_p0 = agent.train_iteration(env, player_idx=0)
        
        # Player 1 iteration
        env.reset()
        utility_p1 = agent.train_iteration(env, player_idx=1)
        
        # Track utilities
        utilities['p0'].append(utility_p0)
        utilities['p1'].append(utility_p1)
        
        # Periodic evaluation
        if iteration % eval_interval == 0 and iteration > 0:
            avg_utility_p0 = np.mean(utilities['p0'][-eval_interval:])
            avg_utility_p1 = np.mean(utilities['p1'][-eval_interval:])
            
            # Estimate exploitability (simplified)
            # In true evaluation, we'd compute best response
            exploitability = abs(avg_utility_p0) + abs(avg_utility_p1)
            exploitabilities.append(exploitability)
            iteration_points.append(iteration)
            
            print(f"\n[Iteration {iteration:,}]")
            print(f"  Avg Utility P0: {avg_utility_p0:.4f}")
            print(f"  Avg Utility P1: {avg_utility_p1:.4f}")
            print(f"  Exploitability: {exploitability:.4f}")
        
        # Save checkpoint
        if iteration % save_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'rlcfr_{iteration}.pt')
            agent.save(checkpoint_path)
    
    # Final save
    final_path = os.path.join(checkpoint_dir, 'rlcfr_final.pt')
    agent.save(final_path)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    
    # Plot results
    plot_training_curves(utilities, exploitabilities, iteration_points, checkpoint_dir)
    
    # Evaluate final strategy
    print("\nEvaluating final strategy...")
    evaluate_strategy(agent, env, n_hands=10000)
    
    return agent


def evaluate_strategy(agent, env, n_hands: int = 10000):
    """
    Evaluate trained RL-CFR strategy.
    
    Args:
        agent: Trained RL-CFR agent
        env: Kuhn poker environment
        n_hands: Number of hands to play
    """
    print(f"\nPlaying {n_hands:,} hands...")
    
    total_reward_p0 = 0
    total_reward_p1 = 0
    
    for _ in tqdm(range(n_hands)):
        state, info = env.reset()
        done = False
        
        while not done:
            current_player = env.current_player
            state_vec = env.get_state_vector(player_idx=current_player)
            legal_actions = env.get_legal_actions()
            
            # Get strategy from average policy
            strategy = agent.get_average_strategy(state_vec, legal_actions)
            
            # Sample action
            action = np.random.choice(len(strategy), p=strategy)
            
            # Step environment
            state, reward, done, truncated, info = env.step(action)
        
        # Track rewards
        if hasattr(env, 'rewards'):
            total_reward_p0 += env.rewards[0]
            total_reward_p1 += env.rewards[1]
    
    avg_reward_p0 = total_reward_p0 / n_hands
    avg_reward_p1 = total_reward_p1 / n_hands
    
    print(f"\nEvaluation Results ({n_hands:,} hands):")
    print(f"  Player 0 avg reward: {avg_reward_p0:.4f}")
    print(f"  Player 1 avg reward: {avg_reward_p1:.4f}")
    print(f"  Sum (should be ~0): {avg_reward_p0 + avg_reward_p1:.4f}")
    
    # For Kuhn poker, optimal strategy should have near-zero exploitability
    # Both players should have ~0 expected value in self-play
    if abs(avg_reward_p0) < 0.05 and abs(avg_reward_p1) < 0.05:
        print("\n✅ Strategy is near Nash equilibrium!")
    else:
        print("\n⚠️  Strategy may not be fully converged yet.")


def plot_training_curves(utilities, exploitabilities, iteration_points, save_dir):
    """Plot training curves."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot utilities
    window = 100
    utilities_p0_smooth = np.convolve(utilities['p0'], np.ones(window)/window, mode='valid')
    utilities_p1_smooth = np.convolve(utilities['p1'], np.ones(window)/window, mode='valid')
    
    axes[0].plot(utilities_p0_smooth, label='Player 0', alpha=0.7)
    axes[0].plot(utilities_p1_smooth, label='Player 1', alpha=0.7)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Utility')
    axes[0].set_title('Training Utilities (smoothed)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot exploitability
    axes[1].plot(iteration_points, exploitabilities, color='red', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Exploitability')
    axes[1].set_title('Exploitability over Training')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n✅ Training curves saved to: {plot_path}")
    plt.close()


def main():
    """Main training entry point."""
    agent = train_kuhn_rlcfr(
        n_iterations=100000,
        save_interval=10000,
        eval_interval=1000,
        checkpoint_dir='checkpoints/kuhn_rlcfr'
    )
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Check training_curves.png for convergence")
    print("2. Load checkpoint and play against the agent")
    print("3. Scale to Texas Hold'em with abstraction")
    print("=" * 70)


if __name__ == "__main__":
    main()
