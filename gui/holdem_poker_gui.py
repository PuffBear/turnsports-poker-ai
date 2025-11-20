import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tkinter as tk
from tkinter import ttk, scrolledtext
import torch
from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.agents.holdem_dqn import DQNAgent
from src.poker.agents.holdem_policy_wrapper import PolicyWrapper
from src.poker.coach.agentic_coach import AgenticCoach

class HoldemPokerGUI:
    def __init__(self, root, agent_path=None):
        self.root = root
        self.root.title("Beating the Boss: Heads-Up No-Limit Hold'em")
        self.root.geometry("1400x800")
        
        # Initialize environment
        self.env = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
        
        # Initialize agent
        self.agent = None
        self.bot_policy = None
        if agent_path and os.path.exists(agent_path):
            self.agent = DQNAgent(state_dim=200, action_dim=9)
            self.agent.load(agent_path)
            self.bot_policy = PolicyWrapper(self.agent)
        
        # Initialize coach
        self.coach = AgenticCoach(bot_policy=self.bot_policy, use_rollouts=True, use_equity=True)
        
        # Player is always player 0 in this setup
        self.player_idx = 0
        self.bot_idx = 1
        
        # Setup UI
        self.setup_ui()
        
        # Start new hand
        self.new_hand()
    
    def setup_ui(self):
        """Create the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Game state
        left_frame = ttk.Frame(main_frame, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(left_frame, text="Heads-Up No-Limit Hold'em", 
                                font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Board display
        ttk.Label(left_frame, text="Board:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky=tk.W)
        self.board_label = ttk.Label(left_frame, text="", font=("Courier", 14))
        self.board_label.grid(row=1, column=1, sticky=tk.W, padx=10)
        
        # Pot
        ttk.Label(left_frame, text="Pot:", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky=tk.W)
        self.pot_label = ttk.Label(left_frame, text="0 BB", font=("Arial", 12))
        self.pot_label.grid(row=2, column=1, sticky=tk.W, padx=10)
        
        # Player info
        ttk.Separator(left_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(left_frame, text="Your Hand:", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky=tk.W)
        self.player_hand_label = ttk.Label(left_frame, text="", font=("Courier", 14, "bold"), foreground="blue")
        self.player_hand_label.grid(row=4, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(left_frame, text="Your Stack:", font=("Arial", 12, "bold")).grid(row=5, column=0, sticky=tk.W)
        self.player_stack_label = ttk.Label(left_frame, text="100 BB", font=("Arial", 12))
        self.player_stack_label.grid(row=5, column=1, sticky=tk.W, padx=10)
        
        # Bot info
        ttk.Separator(left_frame, orient='horizontal').grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(left_frame, text="Bot Stack:", font=("Arial", 12, "bold")).grid(row=7, column=0, sticky=tk.W)
        self.bot_stack_label = ttk.Label(left_frame, text="100 BB", font=("Arial", 12))
        self.bot_stack_label.grid(row=7, column=1, sticky=tk.W, padx=10)
        
        # Action history
        ttk.Label(left_frame, text="Action History:", font=("Arial", 12, "bold")).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        self.history_text = scrolledtext.ScrolledText(left_frame, width=50, height=10, font=("Courier", 10))
        self.history_text.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Action buttons
        ttk.Separator(left_frame, orient='horizontal').grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=11, column=0, columnspan=2, pady=10)
        
        self.action_buttons = {}
        button_configs = [
            (0, "Fold"),
            (1, "Check/Call"),
            (2, "Min Raise"),
            (3, "1/4 Pot"),
            (4, "1/3 Pot"),
            (5, "1/2 Pot"),
            (6, "3/4 Pot"),
            (7, "Pot"),
            (8, "All-In")
        ]
        
        for i, (action_id, label) in enumerate(button_configs):
            btn = ttk.Button(button_frame, text=label, 
                           command=lambda a=action_id: self.player_action(a))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5, sticky=(tk.W, tk.E))
            self.action_buttons[action_id] = btn
        
        # New hand button
        new_hand_btn = ttk.Button(left_frame, text="New Hand", command=self.new_hand)
        new_hand_btn.grid(row=12, column=0, columnspan=2, pady=10)
        
        # Right panel - Coach
        right_frame = ttk.Frame(main_frame, padding="10", relief="solid", borderwidth=2)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(20, 0))
        
        coach_title = ttk.Label(right_frame, text="🤖 AI Coach", font=("Arial", 16, "bold"))
        coach_title.grid(row=0, column=0, pady=10)
        
        # Coach recommendation
        ttk.Label(right_frame, text="Recommendation:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.coach_action_label = ttk.Label(right_frame, text="Waiting...", 
                                           font=("Arial", 14, "bold"), foreground="green")
        self.coach_action_label.grid(row=2, column=0, sticky=tk.W, padx=10)
        
        # Coach explanation
        ttk.Label(right_frame, text="Analysis:", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(15, 5))
        self.coach_text = scrolledtext.ScrolledText(right_frame, width=50, height=25, 
                                                    font=("Courier", 10), wrap=tk.WORD)
        self.coach_text.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Get recommendation button
        get_coach_btn = ttk.Button(right_frame, text="Get Coach Advice", 
                                  command=self.update_coach_recommendation)
        get_coach_btn.grid(row=5, column=0, pady=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def new_hand(self):
        """Start a new hand."""
        self.state, info = self.env.reset()
        self.update_display()
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(tk.END, "New hand started\n")
        
        # If bot is first to act, let it act
        if self.env.current_player == self.bot_idx:
            self.bot_action()
    
    def update_display(self):
        """Update all display elements."""
        # Board
        board_str = ''.join([str(c) for c in self.env.board]) if self.env.board else "---"
        self.board_label.config(text=board_str)
        
        # Pot
        self.pot_label.config(text=f"{self.env.pot:.1f} BB")
        
        # Player hand
        if len(self.env.hands) > self.player_idx:
            hand_str = ''.join([str(c) for c in self.env.hands[self.player_idx]])
            self.player_hand_label.config(text=hand_str)
        
        # Stacks
        if len(self.env.stacks) > self.player_idx:
            self.player_stack_label.config(text=f"{self.env.stacks[self.player_idx]:.1f} BB")
            self.bot_stack_label.config(text=f"{self.env.stacks[self.bot_idx]:.1f} BB")
        
        # Update button states
        legal_actions = self.env._get_legal_actions()
        for action_id, btn in self.action_buttons.items():
            if action_id in legal_actions:
                btn.config(state='normal')
            else:
                btn.config(state='disabled')
    
    def player_action(self, action_id):
        """Handle player action."""
        if self.env.done or self.env.current_player != self.player_idx:
            return
        
        # Execute action
        action_name = AgenticCoach.ACTION_NAMES[action_id]
        self.history_text.insert(tk.END, f"You: {action_name}\n")
        self.history_text.see(tk.END)
        
        self.state, reward, done, truncated, info = self.env.step(action_id)
        self.update_display()
        
        if done:
            self.handle_hand_end(reward, info)
            return
        
        # Bot's turn
        self.root.after(500, self.bot_action)
    
    def bot_action(self):
        """Let bot take action."""
        if self.env.done or self.env.current_player != self.bot_idx:
            return
        
        legal_actions = self.env._get_legal_actions()
        
        if self.bot_policy:
            action = self.bot_policy.get_action(self.state, legal_actions, deterministic=True)
        else:
            # Random fallback
            import random
            action = random.choice(legal_actions)
        
        action_name = AgenticCoach.ACTION_NAMES[action]
        self.history_text.insert(tk.END, f"Bot: {action_name}\n")
        self.history_text.see(tk.END)
        
        self.state, reward, done, truncated, info = self.env.step(action)
        self.update_display()
        
        if done:
            self.handle_hand_end(reward, info)
            return
        
        # Update coach for player's turn
        self.update_coach_recommendation()
    
    def handle_hand_end(self, reward, info):
        """Handle end of hand."""
        winner = info.get('winner', -1)
        
        if winner == self.player_idx:
            msg = f"\n🎉 You won {reward:.1f} BB!\n"
        elif winner == self.bot_idx:
            msg = f"\n😔 Bot won {abs(reward):.1f} BB\n"
        else:
            msg = f"\n🤝 Split pot\n"
        
        self.history_text.insert(tk.END, msg)
        self.history_text.see(tk.END)
        
        # Show hands
        if 'hands' in info:
            self.history_text.insert(tk.END, f"Bot's hand: {''.join([str(c) for c in self.env.hands[self.bot_idx]])}\n")
        
        # Disable buttons
        for btn in self.action_buttons.values():
            btn.config(state='disabled')
    
    def update_coach_recommendation(self):
        """Get and display coach recommendation."""
        if self.env.done or self.env.current_player != self.player_idx:
            self.coach_action_label.config(text="Not your turn")
            self.coach_text.delete(1.0, tk.END)
            return
        
        self.coach_action_label.config(text="Analyzing...")
        self.coach_text.delete(1.0, tk.END)
        self.coach_text.insert(tk.END, "Computing recommendation...\n(This may take a few seconds)")
        self.root.update()
        
        try:
            recommendation = self.coach.get_recommendation(self.env, n_rollouts=100, n_equity_samples=2000)
            
            self.coach_action_label.config(text=recommendation['action_name'])
            
            self.coach_text.delete(1.0, tk.END)
            self.coach_text.insert(tk.END, recommendation['explanation'])
            
        except Exception as e:
            self.coach_action_label.config(text="Error")
            self.coach_text.delete(1.0, tk.END)
            self.coach_text.insert(tk.END, f"Error: {str(e)}")

def main():
    root = tk.Tk()
    
    # Check for trained model
    agent_path = 'checkpoints/dqn_final.pt'
    if not os.path.exists(agent_path):
        print(f"Warning: No trained agent found at {agent_path}")
        print("Bot will play randomly. Train an agent first!")
        agent_path = None
    
    app = HoldemPokerGUI(root, agent_path=agent_path)
    root.mainloop()

if __name__ == "__main__":
    main()
