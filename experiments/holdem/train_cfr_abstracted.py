"""
Training script for Abstracted CFR on Heads-Up No-Limit Hold'em.

This uses card and action abstraction to make CFR tractable:
- Preflop: 169 buckets
- Flop/Turn: 50 buckets each
- River: 10 buckets
- Action space: 5 discrete actions (vs 9 in full game)

Total information sets: ~125,000 (manageable!)

Expected results:
- 10k iterations: ~1 hour, learns basics
- 100k iterations: ~6-12 hours, near-Nash for abstracted game
- Exploitability should decrease monotonically
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from src.poker.agents.cfr_agent import CFRAgent
from src.poker.abstraction import CardAbstraction, ActionAbstraction
from src.poker.envs.holdem_hu_env import HoldemHuEnv


def train_abstracted_cfr(
    n_iterations: int = 50000,
    save_interval: int = 1000,
    eval_interval: int = 100,
    checkpoint_dir: str = 'checkpoints/cfr_abstracted',
    use_action_abstraction: bool = False
):
    """
    Train CFR on abstracted Hold'em.
    
    Args:
        n_iterations: Number of CFR iterations
        save_interval: Save checkpoint every N iterations
        eval_interval: Evaluate and print stats every N iterations
        checkpoint_dir: Directory to save checkpoints
        use_action_abstraction: Whether to use action abstraction (5 actions vs 9)
    
    Returns:
        agent: Trained CFR agent
    """
    print("=" * 80)
    print("Training CFR on Abstracted Heads-Up No-Limit Hold'em")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Iterations: {n_iterations:,}")
    print(f"  Save interval: {save_interval:,}")
    print(f"  Eval interval: {eval_interval:,}")
    print(f"  Action abstraction: {'Enabled (5 actions)' if use_action_abstraction else 'Disabled (9 actions)'}")
    print()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize abstraction
    print("Initializing abstraction...")
    card_abstraction = CardAbstraction()
    
    # Load K-Means models if they exist
    kmeans_flop_path = 'data/kmeans/kmeans_flop_latest.pkl'
    kmeans_turn_path = 'data/kmeans/kmeans_turn_latest.pkl'
    
    if os.path.exists(kmeans_flop_path) and os.path.exists(kmeans_turn_path):
        print("  Loading K-Means models...")
        card_abstraction.load_kmeans_models(kmeans_flop_path, kmeans_turn_path)
    else:
        print("  ⚠️  K-Means models not found - using simple equity bucketing")
        print("     Run scripts/generate_kmeans_clusters.py to generate proper abstractions")
    
    print(f"  Preflop: {card_abstraction.preflop_clusters} buckets")
    print(f"  Flop: {card_abstraction.flop_clusters} buckets")
    print(f"  Turn: {card_abstraction.turn_clusters} buckets")
    print(f"  River: {card_abstraction.river_clusters} buckets")
    
    action_abstraction = None
    num_actions = 9
    if use_action_abstraction:
        action_abstraction = ActionAbstraction()
        num_actions = 5
        print(f"  Actions: {num_actions} discrete actions")
    
    # Initialize environment and agent
    env = HoldemHuEnv()
    agent = CFRAgent(
        card_abstraction=card_abstraction,
        action_abstraction=action_abstraction,
        num_actions=num_actions
    )
    
    print("\nAgent initialized with abstraction!")
    print(f"Expected info sets: ~{169 * 50 * 50 * 10 // 1000}k")
    print()
    
    # Tracking metrics
    utilities = {'p0': [], 'p1': []}
    info_set_counts = []
    iteration_points = []
    
    # Training loop
    print("Starting training...")
    print("-" * 80)
    
    start_time = datetime.now()
    
    for iteration in tqdm(range(n_iterations), desc="CFR Training"):
        # Alternate between players
        # Each player gets updated separately
        
        # Player 0 iteration
        utility_p0 = agent.train_iteration(env, player_idx=0)
        agent.iterations += 1
        
        # Player 1 iteration
        utility_p1 = agent.train_iteration(env, player_idx=1)
        agent.iterations += 1
        
        # Track utilities
        utilities['p0'].append(utility_p0)
        utilities['p1'].append(utility_p1)
        
        # Periodic evaluation
        if iteration % eval_interval == 0 and iteration > 0:
            avg_utility_p0 = np.mean(utilities['p0'][-eval_interval:])
            avg_utility_p1 = np.mean(utilities['p1'][-eval_interval:])
            
            # Count unique information sets
            num_info_sets = len(agent.regret_sum)
            info_set_counts.append(num_info_sets)
            iteration_points.append(iteration)
            
            # Estimate exploitability (simplified)
            # In full evaluation, we'd compute best response
            exploitability = abs(avg_utility_p0) + abs(avg_utility_p1)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            iter_per_sec = iteration / elapsed if elapsed > 0 else 0
            
            print(f"\n[Iteration {iteration:,}] ({elapsed:.0f}s, {iter_per_sec:.1f} iter/s)")
            print(f"  Avg Utility P0: {avg_utility_p0:+.4f}")
            print(f"  Avg Utility P1: {avg_utility_p1:+.4f}")
            print(f"  Exploitability: {exploitability:.4f}")
            print(f"  Unique info sets: {num_info_sets:,}")
        
        # Save checkpoint
        if iteration % save_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'cfr_abstracted_{iteration}.pkl')
            agent.save(checkpoint_path)
            print(f"  ✅ Saved checkpoint to {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(checkpoint_dir, 'cfr_abstracted_final.pkl')
    agent.save(final_path)
    
    elapsed_total = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} minutes)")
    print(f"Final info sets: {len(agent.regret_sum):,}")
    print(f"Model saved to: {final_path}")
    
    # Plot results
    print("\nGenerating training curves...")
    plot_training_curves(utilities, info_set_counts, iteration_points, checkpoint_dir)
    
    # Print abstraction statistics
    print_abstraction_stats(agent)
    
    return agent


def plot_training_curves(utilities, info_set_counts, iteration_points, save_dir):
    """Plot training curves."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot utilities
    window = 100
    if len(utilities['p0']) > window:
        utilities_p0_smooth = np.convolve(utilities['p0'], np.ones(window)/window, mode='valid')
        utilities_p1_smooth = np.convolve(utilities['p1'], np.ones(window)/window, mode='valid')
        
        axes[0, 0].plot(utilities_p0_smooth, label='Player 0', alpha=0.7)
        axes[0, 0].plot(utilities_p1_smooth, label='Player 1', alpha=0.7)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Utility (BB)')
        axes[0, 0].set_title('Training Utilities (smoothed)')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
    
    # Plot info set growth
    if len(info_set_counts) > 0:
        axes[0, 1].plot(iteration_points, info_set_counts, color='green', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Unique Information Sets')
        axes[0, 1].set_title('State Space Coverage')
        axes[0, 1].grid(alpha=0.3)
    
    # Plot utility distribution
    if len(utilities['p0']) > 1000:
        axes[1, 0].hist(utilities['p0'][-50000:], bins=50, alpha=0.7, label='Player 0')
        axes[1, 0].hist(utilities['p1'][-50000:], bins=50, alpha=0.7, label='Player 1')
        axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel('Utility (BB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Utility Distribution (last 10k)')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # Plot exploitability estimate
    if len(utilities['p0']) > window:
        exploitability = []
        for i in range(window, len(utilities['p0']), window):
            exp = abs(np.mean(utilities['p0'][i-window:i])) + abs(np.mean(utilities['p1'][i-window:i]))
            exploitability.append(exp)
        
        axes[1, 1].plot(range(window, len(utilities['p0']), window), exploitability, 
                       color='red', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Exploitability (approx)')
        axes[1, 1].set_title('Exploitability over Training')
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'training_curves_50k.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Training curves saved to: {plot_path}")
    plt.close()


def print_abstraction_stats(agent):
    """Print statistics about the learned strategy."""
    print("\n" + "=" * 80)
    print("Abstraction Statistics")
    print("=" * 80)
    
    # Count info sets by street
    street_counts = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0}
    
    for info_set in agent.regret_sum.keys():
        if 'preflop' in info_set:
            street_counts['preflop'] += 1
        elif 'flop' in info_set:
            street_counts['flop'] += 1
        elif 'turn' in info_set:
            street_counts['turn'] += 1
        elif 'river' in info_set:
            street_counts['river'] += 1
    
    print("\nInformation sets visited by street:")
    for street, count in street_counts.items():
        print(f"  {street.capitalize()}: {count:,}")
    
    total = sum(street_counts.values())
    print(f"\nTotal unique info sets: {total:,}")
    print(f"Memory usage (approx): {total * 8 * agent.num_actions * 2 / 1024 / 1024:.1f} MB")
    
    # Sample some strategies
    print("\nSample strategies (first 5 info sets):")
    for i, info_set in enumerate(list(agent.strategy_sum.keys())[:5]):
        avg_strategy = agent.strategy_sum[info_set] / (agent.strategy_sum[info_set].sum() + 1e-10)
        print(f"  {info_set}: {np.around(avg_strategy, 3)}")
    
    print()


def main():
    """Main training entry point."""
    
    # Quick test (10k iterations)
    # agent = train_abstracted_cfr(
    #     n_iterations=50000,
    #     save_interval=5000,
    #     eval_interval=500,
    #     checkpoint_dir='checkpoints/cfr_abstracted_test',
    #     use_action_abstraction=False
    # )
    
    # Full training (100k iterations)
    agent = train_abstracted_cfr(
        n_iterations=500000,
        save_interval=50000,
        eval_interval=5000,
        checkpoint_dir='checkpoints/cfr_abstracted',
        use_action_abstraction=True  # ENABLED: 5 actions instead of 9 for faster training
    )
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("1. Check training_curves.png for convergence")
    print("2. Evaluate against baseline opponents (Random, TAG, LAG)")
    print("3. Integrate with GUI for human play")
    print("4. Compare to DQN performance (CFR should dominate)")
    print("=" * 80)


if __name__ == "__main__":
    main()
