#!/usr/bin/env python3
"""
Command-line interface for playing poker with the AI coach.
No GUI required - works in terminal!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.opponents.holdem_rule_based import RandomOpponent, TAGOpponent
from src.poker.coach.agentic_coach import AgenticCoach
from src.poker.agents.holdem_dqn import DQNAgent
from src.poker.agents.holdem_policy_wrapper import PolicyWrapper

class PokerCLI:
    def __init__(self, use_bot=False):
        self.env = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
        
        # Load bot if available
        self.bot_policy = None
        if use_bot and os.path.exists('checkpoints/dqn_final.pt'):
            print("Loading trained bot...")
            agent = DQNAgent(state_dim=200, action_dim=9)
            agent.load('checkpoints/dqn_final.pt')
            self.bot_policy = PolicyWrapper(agent)
            print("✓ Bot loaded\n")
        else:
            print("Using random opponent\n")
        
        # Initialize coach
        self.coach = AgenticCoach(
            bot_policy=self.bot_policy,
            use_rollouts=self.bot_policy is not None,
            use_equity=True
        )
        
        self.player_idx = 0
        self.bot_idx = 1
        
    def display_state(self):
        """Display current game state."""
        print("\n" + "=" * 60)
        
        # Board
        board_str = ' '.join([str(c) for c in self.env.board]) if self.env.board else "---"
        street_names = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
        print(f"Street: {street_names[self.env.street]}")
        print(f"Board: {board_str}")
        print(f"Pot: {self.env.pot:.1f} BB")
        print()
        
        # Your hand
        hand_str = ' '.join([str(c) for c in self.env.hands[self.player_idx]])
        print(f"Your Hand: {hand_str}")
        print(f"Your Stack: {self.env.stacks[self.player_idx]:.1f} BB")
        print(f"Bot Stack: {self.env.stacks[self.bot_idx]:.1f} BB")
        
        # To call
        to_call = self.env.get_to_call() if hasattr(self.env, "get_to_call") else abs(self.env.street_investment[0] - self.env.street_investment[1])
        if to_call > 0:
            print(f"\nTo Call: {to_call:.1f} BB")
        
        print("=" * 60)
    
    def get_action_from_user(self):
        """Get action from user input."""
        legal_actions = self.env._get_legal_actions()
        
        print("\nLegal Actions:")
        action_map = {}
        for i, action_id in enumerate(legal_actions):
            name = AgenticCoach.ACTION_NAMES[action_id]
            action_map[str(i+1)] = action_id
            print(f"  {i+1}. {name}")
        
        # Add coach option
        print(f"  C. Get Coach Advice")
        
        while True:
            choice = input("\nYour choice (1-9 or C): ").strip().upper()
            
            if choice == 'C':
                self.show_coach_advice()
                continue
                
            if choice in action_map:
                return action_map[choice]
            
            print("Invalid choice. Try again.")
    
    def show_coach_advice(self):
        """Display coach recommendation."""
        print("\n" + "-" * 60)
        print("🤖 COACH ANALYSIS")
        print("-" * 60)
        
        try:
            recommendation = self.coach.get_recommendation(
                self.env,
                n_rollouts=100 if self.bot_policy else 0,
                n_equity_samples=2000
            )
            
            print(f"\nRecommended: {recommendation['action_name']}")
            print("\n" + recommendation['explanation'])
            
        except Exception as e:
            print(f"Error getting coach advice: {e}")
        
        print("-" * 60)
    
    def bot_action(self):
        """Let bot take action."""
        legal_actions = self.env._get_legal_actions()
        
        if self.bot_policy:
            action = self.bot_policy.get_action(
                self.env._get_obs(),
                legal_actions,
                deterministic=True
            )
        else:
            # Random opponent
            opponent = RandomOpponent()
            action = opponent.get_action(self.env, self.env._get_obs())
        
        action_name = AgenticCoach.ACTION_NAMES[action]
        print(f"\nBot action: {action_name}")
        
        return action
    
    def play_hand(self):
        """Play one hand."""
        state, info = self.env.reset()
        
        print("\n\n" + "=" * 60)
        print("NEW HAND")
        print("=" * 60)
        
        done = False
        
        while not done:
            self.display_state()
            
            if self.env.current_player == self.player_idx:
                # Your turn
                print("\n>> YOUR TURN <<")
                action = self.get_action_from_user()
            else:
                # Bot's turn
                print("\n>> BOT'S TURN <<")
                input("Press Enter to see bot action...")
                action = self.bot_action()
            
            # Execute action
            state, reward, done, truncated, info = self.env.step(action)
        
        # Hand finished
        self.display_state()
        print("\n" + "=" * 60)
        print("HAND RESULT")
        print("=" * 60)
        
        winner = info.get('winner', -1)
        if winner == self.player_idx:
            print(f"🎉 YOU WON {reward:.1f} BB!")
        elif winner == self.bot_idx:
            print(f"😔 Bot won {abs(reward):.1f} BB")
        else:
            print("🤝 Split pot")
        
        # Show bot's hand
        bot_hand = ' '.join([str(c) for c in self.env.hands[self.bot_idx]])
        print(f"\nBot's hand: {bot_hand}")
        print(f"Final board: {' '.join([str(c) for c in self.env.board])}")
        print("=" * 60)
    
    def run(self):
        """Main game loop."""
        print("\n" + "=" * 60)
        print("POKER CLI - PLAY WITH AI COACH")
        print("=" * 60)
        print("\nCommands:")
        print("  1-9: Choose action")
        print("  C:   Get coach advice (shows equity, EV, strategy)")
        print("  (Coach analyzes your hand and recommends best play)")
        print("\n" + "=" * 60)
        
        while True:
            self.play_hand()
            
            play_again = input("\nPlay another hand? (y/n): ").strip().lower()
            if play_again != 'y':
                print("\nThanks for playing!")
                break

def main():
    # Check for trained bot
    use_bot = os.path.exists('checkpoints/dqn_final.pt')
    
    if not use_bot:
        print("\nNo trained bot found. Playing vs random opponent.")
        print("(Train bot with: python experiments/holdem/train_holdem_dqn_vs_pool.py)\n")
    
    cli = PokerCLI(use_bot=use_bot)
    cli.run()

if __name__ == "__main__":
    main()
