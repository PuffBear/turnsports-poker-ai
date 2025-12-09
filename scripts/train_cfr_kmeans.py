#!/usr/bin/env python3
"""
Quick CFR training script with K-Means abstraction.
Optimized for faster convergence with meaningful clusters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.holdem.train_cfr_abstracted import train_abstracted_cfr

if __name__ == "__main__":
    print("=" * 80)
    print("CFR Training with K-Means Card Abstraction")
    print("=" * 80)
    print()
    print("This will train a poker bot using:")
    print("  ✅ K-Means equity distribution clustering (meaningful hand buckets)")
    print("  ✅ Action abstraction (5 discrete bet sizes)")
    print("  ✅ Counterfactual Regret Minimization (CFR)")
    print()
    print("Training configuration:")
    print("  - Iterations: 25,000 (approx 1.5-2 hours)")
    print("  - Save every: 5,000 iterations")
    print("  - Evaluate every: 500 iterations")
    print()
    print("Expected outcome:")
    print("  - Bot learns solid strategy (good for competitive play)")
    print("  - Better than 5k baseline, not quite Nash equilibrium")
    print("  - Should beat most casual players")
    print()
    
    response = input("Start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    print()
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()
    
    # Train with good defaults
    agent = train_abstracted_cfr(
        n_iterations=25000,         # 25k iterations
        save_interval=5000,         # Save every 5k
        eval_interval=500,          # Eval every 500
        checkpoint_dir='checkpoints/cfr_kmeans',
        use_action_abstraction=True  # 5 actions for faster convergence
    )
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("Your trained bot is saved in: checkpoints/cfr_kmeans/")
    print()
    print("Next steps:")
    print("  1. Play against it: venv/bin/python gui/cfr_poker_gui.py")
    print("     (Update the checkpoint path to 'cfr_kmeans/cfr_abstracted_final.pkl')")
    print("  2. Check training curves: checkpoints/cfr_kmeans/training_curves_50k.png")
    print("  3. Compare to your old random-bucketing bot!")
    print()
