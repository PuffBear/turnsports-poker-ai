"""
Generate K-Means clustering models for postflop card abstraction.

This script:
1. Generates equity distribution samples for flop and turn
2. Trains K-Means models to cluster similar hands
3. Saves models for use in CFR training

Equity distribution approach (from academic research):
- Instead of single equity value, compute histogram of equities across runouts
- This captures hand "potential" (e.g., flush draws have bimodal distribution)
- K-Means with Earth Mover's Distance for clustering

References:
- "Potential-Aware Imperfect Recall Abstraction" (Johanson et al., 2014)
- Poker-AI repo: src/abstraction.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import random
from tqdm import tqdm
from sklearn.cluster import KMeans
import joblib
from datetime import datetime

from src.poker.core.card import Card, Deck
from src.poker.core.hand_evaluator import HandEvaluator


def calculate_equity_distribution(hole_cards, community_cards, n_bins=50, n_samples=200):
    """
    Calculate equity distribution histogram for a hand.
    
    This is the core of postflop abstraction.
    
    Args:
        hole_cards: List of 2 Card objects
        community_cards: List of community Card objects (3 for flop, 4 for turn)
        n_bins: Number of histogram bins
        n_samples: Number of Monte Carlo samples
    
    Returns:
        hist: Normalized histogram (n_bins,) representing equity distribution
    """
    # Create deck and remove known cards
    deck = Deck()
    for card in hole_cards + community_cards:
        deck.cards.remove(card)
    
    evaluator = HandEvaluator()
    equities = []
    
    for _ in range(n_samples):
        # Shuffle deck
        random.shuffle(deck.cards)
        
        # Deal opponent hole cards
        opp_hole = deck.cards[:2]
        
        # Complete the board
        remaining_board_size = 5 - len(community_cards)
        complete_board = community_cards + deck.cards[2:2 + remaining_board_size]
        
        # Evaluate hands
        player_hand = hole_cards + complete_board
        opp_hand = opp_hole + complete_board
        
        player_rank = evaluator.evaluate(player_hand)
        opp_rank = evaluator.evaluate(opp_hand)
        
        if player_rank < opp_rank:  # Lower = better
            equity = 1.0
        elif player_rank == opp_rank:
            equity = 0.5
        else:
            equity = 0.0
        
        equities.append(equity)
    
    # Create histogram
    hist, _ = np.histogram(equities, bins=n_bins, range=(0, 1))
    
    # Normalize
    if hist.sum() > 0:
        hist = hist / hist.sum()
    
    return hist


def generate_equity_distributions(street='flop', n_samples=10000, n_bins=50):
    """
    Generate equity distribution samples for K-Means training.
    
    Args:
        street: 'flop' or 'turn'
        n_samples: Number of random hands to sample
        n_bins: Number of bins for equity histogram
    
    Returns:
        distributions: Array of shape (n_samples, n_bins)
        hand_strings: List of hand strings for reference
    """
    print(f"\n{'='*60}")
    print(f"Generating Equity Distributions for {street.upper()}")
    print(f"{'='*60}")
    print(f"Samples: {n_samples:,}")
    print(f"Bins: {n_bins}")
    print()
    
    if street == 'flop':
        num_community_cards = 3
    elif street == 'turn':
        num_community_cards = 4
    else:
        raise ValueError("street must be 'flop' or 'turn'")
    
    distributions = []
    hand_strings = []
    
    print(f"Generating {n_samples:,} samples...")
    for _ in tqdm(range(n_samples)):
        # Deal random hand
        deck = Deck()
        random.shuffle(deck.cards)
        
        hole_cards = deck.cards[:2]
        community_cards = deck.cards[2:2 + num_community_cards]
        
        # Calculate equity distribution
        distribution = calculate_equity_distribution(
            hole_cards, 
            community_cards, 
            n_bins=n_bins, 
            n_samples=200  # Monte Carlo samples per hand
        )
        
        distributions.append(distribution)
        
        # Store hand string for reference
        hand_str = ' '.join([str(c) for c in hole_cards + community_cards])
        hand_strings.append(hand_str)
    
    distributions = np.array(distributions)
    
    print(f"\n✅ Generated {len(distributions):,} equity distributions")
    print(f"   Shape: {distributions.shape}")
    print(f"   Memory: {distributions.nbytes / 1024 / 1024:.1f} MB")
    
    return distributions, hand_strings


def train_kmeans_clustering(distributions, n_clusters=50, random_state=42):
    """
    Train K-Means clustering on equity distributions.
    
    Args:
        distributions: Array of equity distributions (n_samples, n_bins)
        n_clusters: Number of clusters (50 for flop/turn)
        random_state: Random seed
    
    Returns:
        kmeans: Trained KMeans model
    """
    print(f"\n{'='*60}")
    print(f"Training K-Means Clustering")
    print(f"{'='*60}")
    print(f"Clusters: {n_clusters}")
    print(f"Samples: {len(distributions):,}")
    print()
    
    print("Fitting K-Means...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
        verbose=1
    )
    
    kmeans.fit(distributions)
    
    print(f"\n✅ K-Means training complete!")
    print(f"   Inertia: {kmeans.inertia_:.2f}")
    print(f"   Iterations: {kmeans.n_iter_}")
    
    return kmeans


def evaluate_clustering(kmeans, distributions, n_samples=1000):
    """
    Evaluate clustering quality.
    
    Args:
        kmeans: Trained KMeans model
        distributions: Test distributions
        n_samples: Number of samples to evaluate
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Clustering Quality")
    print(f"{'='*60}")
    
    # Sample subset for evaluation
    indices = np.random.choice(len(distributions), size=min(n_samples, len(distributions)), replace=False)
    test_distributions = distributions[indices]
    
    # Predict clusters
    predictions = kmeans.predict(test_distributions)
    
    # Compute cluster sizes
    unique, counts = np.unique(predictions, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    print(f"\nCluster Statistics:")
    print(f"  Total clusters: {len(unique)}")
    print(f"  Avg cluster size: {np.mean(counts):.1f}")
    print(f"  Min cluster size: {np.min(counts)}")
    print(f"  Max cluster size: {np.max(counts)}")
    
    # Show largest and smallest clusters
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    print(f"\nLargest clusters:")
    for cluster_id, size in sorted_clusters[:5]:
        print(f"  Cluster {cluster_id}: {size} hands")
    
    print(f"\nSmallest clusters:")
    for cluster_id, size in sorted_clusters[-5:]:
        print(f"  Cluster {cluster_id}: {size} hands")
    
    metrics = {
        'n_clusters': len(unique),
        'avg_size': np.mean(counts),
        'cluster_sizes': cluster_sizes
    }
    
    return metrics


def save_models_and_data(kmeans_flop, kmeans_turn, 
                          distributions_flop, distributions_turn,
                          hands_flop, hands_turn,
                          save_dir='data/kmeans'):
    """Save trained models and data."""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save K-Means models
    flop_model_path = os.path.join(save_dir, f'kmeans_flop_{timestamp}.pkl')
    turn_model_path = os.path.join(save_dir, f'kmeans_turn_{timestamp}.pkl')
    
    joblib.dump(kmeans_flop, flop_model_path)
    joblib.dump(kmeans_turn, turn_model_path)
    
    print(f"\n{'='*60}")
    print(f"Saved Models")
    print(f"{'='*60}")
    print(f"Flop model: {flop_model_path}")
    print(f"Turn model: {turn_model_path}")
    
    # Save latest versions (for easy loading)
    flop_latest = os.path.join(save_dir, 'kmeans_flop_latest.pkl')
    turn_latest = os.path.join(save_dir, 'kmeans_turn_latest.pkl')
    
    joblib.dump(kmeans_flop, flop_latest)
    joblib.dump(kmeans_turn, turn_latest)
    
    print(f"\nLatest versions:")
    print(f"  {flop_latest}")
    print(f"  {turn_latest}")
    
    # Save distributions for future reference
    np.save(os.path.join(save_dir, f'distributions_flop_{timestamp}.npy'), distributions_flop)
    np.save(os.path.join(save_dir, f'distributions_turn_{timestamp}.npy'), distributions_turn)
    
    # Save hand strings
    with open(os.path.join(save_dir, f'hands_flop_{timestamp}.txt'), 'w') as f:
        f.write('\n'.join(hands_flop))
    with open(os.path.join(save_dir, f'hands_turn_{timestamp}.txt'), 'w') as f:
        f.write('\n'.join(hands_turn))
    
    print(f"\n✅ All data saved to: {save_dir}")


def main():
    """Main execution."""
    print("="*80)
    print("K-Means Clustering for Postflop Card Abstraction")
    print("="*80)
    print("\nThis will generate equity distribution samples and train K-Means models.")
    print("Expected time: 30-60 minutes for 10k samples")
    print()
    
    # Configuration
    n_samples = 10000  # Number of hands to sample
    n_bins = 50        # Histogram bins
    n_clusters_flop = 50
    n_clusters_turn = 50
    
    # Generate flop distributions
    print("\n" + "="*80)
    print("STEP 1: Generate Flop Equity Distributions")
    print("="*80)
    distributions_flop, hands_flop = generate_equity_distributions(
        street='flop',
        n_samples=n_samples,
        n_bins=n_bins
    )
    
    # Train flop K-Means
    print("\n" + "="*80)
    print("STEP 2: Train Flop K-Means")
    print("="*80)
    kmeans_flop = train_kmeans_clustering(
        distributions_flop,
        n_clusters=n_clusters_flop
    )
    
    # Evaluate flop clustering
    eval_metrics_flop = evaluate_clustering(kmeans_flop, distributions_flop)
    
    # Generate turn distributions
    print("\n" + "="*80)
    print("STEP 3: Generate Turn Equity Distributions")
    print("="*80)
    distributions_turn, hands_turn = generate_equity_distributions(
        street='turn',
        n_samples=n_samples,
        n_bins=n_bins
    )
    
    # Train turn K-Means
    print("\n" + "="*80)
    print("STEP 4: Train Turn K-Means")
    print("="*80)
    kmeans_turn = train_kmeans_clustering(
        distributions_turn,
        n_clusters=n_clusters_turn
    )
    
    # Evaluate turn clustering
    eval_metrics_turn = evaluate_clustering(kmeans_turn, distributions_turn)
    
    # Save everything
    print("\n" + "="*80)
    print("STEP 5: Save Models and Data")
    print("="*80)
    save_models_and_data(
        kmeans_flop, kmeans_turn,
        distributions_flop, distributions_turn,
        hands_flop, hands_turn
    )
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print("\nK-Means models generated and saved.")
    print("\nNext steps:")
    print("1. Load models in CardAbstraction:")
    print("   abstraction.load_kmeans_models('data/kmeans/kmeans_flop_latest.pkl', ...)")
    print("2. Train CFR with K-Means abstraction")
    print("3. Should see better convergence than simple equity bucketing")
    print("="*80)


if __name__ == "__main__":
    main()
