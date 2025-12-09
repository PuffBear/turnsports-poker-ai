#!/usr/bin/env python3
"""
Script to add LLM Coach to cfr_poker_gui.py

Run with: python3 add_coach_to_gui.py
"""

import os
import re

def main():
    gui_file = 'gui/cfr_poker_gui.py'
    backup_file = 'gui/cfr_poker_gui.py.backup'
    
    # Create backup
    if not os.path.exists(backup_file):
        with open(gui_file, 'r') as f:
            content = f.read()
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"✅ Created backup: {backup_file}")
    
    print("\n" + "="*60)
    print("LLM Coach GUI Integration Script")
    print("="*60)
    print("\nThis script will add coach recommendations panel to your GUI.")
    print("The new GUI will have:")
    print("  • AI-powered poker coach recommendations")
    print("  • Risk tolerance controls (conservative/moderate/aggressive)")
    print("  • Real-time equity and pot odds analysis")
    print("  • Multi-level strategy suggestions")
    print("\nOriginal file backed up to: gui/cfr_poker_gui.py.backup")
    print("\nTo complete the integration manually:")
    print("\n1. Add this import after line 14:")
    print("   from src.poker.coach.llm_coach import LLMCoach")
    print("\n2. Change window size (line 20):")
    print("   self.root.geometry('1600x900')  # Wider for recommendation panel")
    print("\n3. Add coach initialization after session_stats (line ~67):")
    print('''
        # Initialize LLM Coach
        try:
            bot_policy = None
            self.llm_coach = LLMCoach(bot_policy=bot_policy, risk_tolerance='moderate')
            print("✅ LLM Coach initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize coach: {e}")
            self.llm_coach = None
''')
    print("\n4. Replace your current GUI file with the example from:")
    print("   gui/cfr_poker_gui_example_with_coach.py")
    print("\nOr copy the full implementation from the other branch:")
    print("   git checkout origin/feature/llm-coach-clean -- gui/cfr_poker_gui.py")
    print("\n" + "="*60)
    print("\nRecommendation: Use the version from the llm-coach-clean branch")
    print("Run: git checkout origin/feature/llm-coach-clean -- gui/cfr_poker_gui.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
