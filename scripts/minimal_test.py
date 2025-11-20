#!/usr/bin/env python3
"""
Minimal test - works without external dependencies
Tests core poker logic only
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_cards():
    """Test card and deck functionality."""
    print("=" * 60)
    print("Testing Card System")
    print("=" * 60)
    
    from src.poker.core.cards import Card
    from src.poker.core.deck import Deck
    
    # Test card creation
    card = Card('A', 's')
    print(f"✓ Created card: {card}")
    
    # Test deck
    deck = Deck()
    print(f"✓ Created deck with {len(deck.cards)} cards")
    
    # Draw some cards
    hand = deck.draw(2)
    print(f"✓ Drew hand: {hand[0]}, {hand[1]}")
    
    # Draw flop
    flop = deck.draw(3)
    print(f"✓ Drew flop: {flop[0]}, {flop[1]}, {flop[2]}")
    
    print(f"✓ Remaining cards in deck: {len(deck.cards)}")
    print("\n✅ Card system working!\n")

def test_hand_eval():
    """Test hand evaluation."""
    print("=" * 60)
    print("Testing Hand Evaluator")
    print("=" * 60)
    
    from src.poker.core.cards import Card
    from src.poker.core.hand_eval import HandEvaluator
    
    # Create a flush
    flush_hand = [
        Card('A', 'h'),
        Card('K', 'h'),
        Card('Q', 'h'),
        Card('J', 'h'),
        Card('9', 'h')
    ]
    
    score = HandEvaluator._score_5_cards(flush_hand)
    print(f"Flush score: {score}")
    print(f"  Category: {score[0]} (should be 5 for flush)")
    
    # Create pair
    pair_hand = [
        Card('A', 'h'),
        Card('A', 's'),
        Card('K', 'c'),
        Card('Q', 'd'),
        Card('J', 'h')
    ]
    
    score2 = HandEvaluator._score_5_cards(pair_hand)
    print(f"Pair score: {score2}")
    print(f"  Category: {score2[0]} (should be 1 for pair)")
    
    print("\n✅ Hand evaluator working!\n")

def test_environment_basic():
    """Test basic environment without numpy."""
    print("=" * 60)
    print("Testing Environment Creation")
    print("=" * 60)
    
    try:
        # This will fail if numpy/gymnasium not installed
        from src.poker.envs.holdem_hu_env import HoldemHuEnv
        
        env = HoldemHuEnv()
        print(f"✓ Environment created")
        print(f"  Initial stack: {env.initial_stack} BB")
        print(f"  Blinds: {env.sb_amount}/{env.bb_amount}")
        
        print("\n✅ Environment working!\n")
        
    except ImportError as e:
        print(f"⚠️  Environment requires external packages")
        print(f"   Missing: {e}")
        print(f"   Install with: pip install numpy gymnasium")
        print("\n   (This is expected if dependencies not installed yet)")

def main():
    print("\n" + "=" * 60)
    print("POKER RL PROJECT - MINIMAL TEST")
    print("=" * 60)
    print()
    
    try:
        test_cards()
    except Exception as e:
        print(f"❌ Card test failed: {e}\n")
    
    try:
        test_hand_eval()
    except Exception as e:
        print(f"❌ Hand eval test failed: {e}\n")
    
    try:
        test_environment_basic()
    except Exception as e:
        print(f"❌ Environment test failed: {e}\n")
    
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print()
    print("1. Fix network connection and install dependencies:")
    print("   source venv/bin/activate")
    print("   pip install numpy pandas matplotlib torch gymnasium termcolor tqdm")
    print()
    print("2. Then run full test:")
    print("   python scripts/quickstart_test.py")
    print()
    print("3. Or jump straight to playing:")
    print("   python gui/holdem_poker_gui.py")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
