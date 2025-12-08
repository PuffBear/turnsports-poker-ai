import sys
import os
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.agents.cfr_agent import CFRAgent
from src.poker.abstraction.card_abstraction import CardAbstraction
from src.poker.abstraction.action_abstraction import ActionAbstraction
from src.poker.coach.llm_coach import LLMCoach

class CFRPokerGUI:
    def __init__(self, root, checkpoint_path=None):
        self.root = root
        self.root.title("Heads-Up No-Limit Hold'em: You vs CFR Bot")
        self.root.geometry("1600x900")  # Wider for recommendation window
        
        # Initialize environment
        self.env = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
        
        # Initialize abstractions
        self.card_abstraction = CardAbstraction()
        # Note: We are using the default simple equity bucketing for now as we don't have K-Means paths
        
        self.action_abstraction = ActionAbstraction()
        
        # Initialize agent
        self.agent = CFRAgent(
            card_abstraction=self.card_abstraction,
            action_abstraction=self.action_abstraction,
            num_actions=5 # Abstracted actions
        )
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            self.agent.load(checkpoint_path)
            print("Checkpoint loaded.")
        else:
            print("Warning: No checkpoint found. Bot will play randomly/uniformly.")
        
        # Player is always player 0 in this setup for GUI convenience
        # But in env, button rotates. We'll handle this by mapping GUI player to env player.
        # Actually, let's keep it simple: Human is always P0, Bot is P1.
        # We might need to force positions or handle button rotation.
        # The env randomizes button in reset().
        self.human_idx = 0
        self.bot_idx = 1
        
        # Initialize LLM Coach (will use template-based if no API key)
        # Note: We skip loading DQN policy to avoid PyTorch dependency issues
        # The coach will still work with equity calculations and pot odds
        try:
            bot_policy = None  # Skip DQN policy loading for now
            self.llm_coach = LLMCoach(bot_policy=bot_policy, risk_tolerance='moderate')
            print("LLM Coach initialized (without rollouts - will use equity/pot odds analysis)")
        except Exception as e:
            print(f"Warning: Could not initialize LLM coach: {e}")
            print("Falling back to template-based recommendations only.")
            self.llm_coach = None
        
        # Setup UI
        self.setup_ui()
        
        # Start new hand
        self.new_hand()
    
    def setup_ui(self):
        """Create the GUI layout."""
        # Main container - split into game and recommendation panels
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Game panel (left side)
        game_panel = ttk.Frame(main_frame, padding="5")
        game_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Recommendation panel (right side)
        self.setup_recommendation_panel(main_frame)
        
        # Game Info Frame
        info_frame = ttk.LabelFrame(game_panel, text="Game Info", padding="10")
        info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Board
        ttk.Label(info_frame, text="Board:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W)
        self.board_label = ttk.Label(info_frame, text="---", font=("Courier", 14))
        self.board_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        # Pot
        ttk.Label(info_frame, text="Pot:", font=("Arial", 12, "bold")).grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.pot_label = ttk.Label(info_frame, text="0 BB", font=("Arial", 12))
        self.pot_label.grid(row=0, column=3, sticky=tk.W, padx=10)
        
        # Street
        ttk.Label(info_frame, text="Street:", font=("Arial", 12, "bold")).grid(row=0, column=4, sticky=tk.W, padx=(20, 0))
        self.street_label = ttk.Label(info_frame, text="PREFLOP", font=("Arial", 12))
        self.street_label.grid(row=0, column=5, sticky=tk.W, padx=10)

        # Player Frame
        player_frame = ttk.LabelFrame(game_panel, text="You (Hero)", padding="10")
        player_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(0, 5))
        
        ttk.Label(player_frame, text="Hand:", font=("Arial", 12)).grid(row=0, column=0, sticky=tk.W)
        self.player_hand_label = ttk.Label(player_frame, text="?? ??", font=("Courier", 14, "bold"), foreground="blue")
        self.player_hand_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(player_frame, text="Stack:", font=("Arial", 12)).grid(row=1, column=0, sticky=tk.W)
        self.player_stack_label = ttk.Label(player_frame, text="100.0 BB", font=("Arial", 12))
        self.player_stack_label.grid(row=1, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(player_frame, text="Role:", font=("Arial", 12)).grid(row=2, column=0, sticky=tk.W)
        self.player_role_label = ttk.Label(player_frame, text="---", font=("Arial", 12))
        self.player_role_label.grid(row=2, column=1, sticky=tk.W, padx=10)

        # Bot Frame
        bot_frame = ttk.LabelFrame(game_panel, text="Bot (Villain)", padding="10")
        bot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(5, 0))
        
        ttk.Label(bot_frame, text="Hand:", font=("Arial", 12)).grid(row=0, column=0, sticky=tk.W)
        self.bot_hand_label = ttk.Label(bot_frame, text="?? ??", font=("Courier", 14, "bold"))
        self.bot_hand_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(bot_frame, text="Stack:", font=("Arial", 12)).grid(row=1, column=0, sticky=tk.W)
        self.bot_stack_label = ttk.Label(bot_frame, text="100.0 BB", font=("Arial", 12))
        self.bot_stack_label.grid(row=1, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(bot_frame, text="Role:", font=("Arial", 12)).grid(row=2, column=0, sticky=tk.W)
        self.bot_role_label = ttk.Label(bot_frame, text="---", font=("Arial", 12))
        self.bot_role_label.grid(row=2, column=1, sticky=tk.W, padx=10)

        # Action History
        hist_frame = ttk.LabelFrame(game_panel, text="Action History", padding="10")
        hist_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.history_text = scrolledtext.ScrolledText(hist_frame, width=60, height=10, font=("Courier", 10))
        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Controls
        control_frame = ttk.Frame(game_panel, padding="10")
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.action_buttons = {}
        actions = [
            (0, "Fold"), (1, "Check/Call"), (2, "Min Raise"),
            (3, "1/4 Pot"), (4, "1/3 Pot"), (5, "1/2 Pot"),
            (6, "3/4 Pot"), (7, "Pot"), (8, "All-In")
        ]
        
        for i, (aid, name) in enumerate(actions):
            btn = ttk.Button(control_frame, text=name, command=lambda a=aid: self.human_action(a))
            btn.grid(row=0, column=i, padx=2)
            self.action_buttons[aid] = btn
            
        # New Hand Button
        ttk.Button(game_panel, text="New Hand", command=self.new_hand).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)  # Game panel gets 2/3
        main_frame.columnconfigure(1, weight=1)  # Recommendation panel gets 1/3
        game_panel.columnconfigure(0, weight=1)
        game_panel.columnconfigure(1, weight=1)

    def setup_recommendation_panel(self, parent):
        """Create the LLM recommendation panel."""
        rec_frame = ttk.LabelFrame(parent, text="AI Coach Recommendations", padding="10")
        rec_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Risk tolerance selector
        risk_frame = ttk.Frame(rec_frame)
        risk_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(risk_frame, text="Risk Tolerance:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.risk_var = tk.StringVar(value="moderate")
        for risk in ["conservative", "moderate", "aggressive"]:
            rb = ttk.Radiobutton(risk_frame, text=risk.capitalize(), variable=self.risk_var, 
                                value=risk, command=self.on_risk_change)
            rb.pack(side=tk.LEFT, padx=5)
        
        # Stats display
        stats_frame = ttk.LabelFrame(rec_frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=8, width=35, font=("Courier", 9), wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        stats_scroll = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=stats_scroll.set)
        
        # Recommendation display
        rec_display_frame = ttk.LabelFrame(rec_frame, text="Recommendation", padding="5")
        rec_display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.rec_text = scrolledtext.ScrolledText(rec_display_frame, height=15, width=35, 
                                                   font=("Arial", 10), wrap=tk.WORD)
        self.rec_text.pack(fill=tk.BOTH, expand=True)
        
        # Risk analysis display
        risk_analysis_frame = ttk.LabelFrame(rec_frame, text="All Risk Levels", padding="5")
        risk_analysis_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        
        self.risk_analysis_text = tk.Text(risk_analysis_frame, height=6, width=35, 
                                         font=("Courier", 9), wrap=tk.WORD)
        self.risk_analysis_text.pack(fill=tk.BOTH, expand=True)
    
    def on_risk_change(self):
        """Handle risk tolerance change."""
        if self.llm_coach:
            self.llm_coach.set_risk_tolerance(self.risk_var.get())
            self.update_recommendations()
    
    def update_recommendations(self):
        """Update the recommendation panel with current analysis."""
        if not self.llm_coach or self.env.done or self.env.current_player != self.human_idx:
            # Clear or show "Not your turn" message
            if self.env.done:
                self.stats_text.delete(1.0, tk.END)
                self.rec_text.delete(1.0, tk.END)
                self.risk_analysis_text.delete(1.0, tk.END)
            return
        
        try:
            # Get recommendation
            rec = self.llm_coach.get_recommendation(self.env, n_rollouts=100, n_equity_samples=1000)
            stats = rec.get('stats', {})
            
            # Update stats display
            self.stats_text.delete(1.0, tk.END)
            stats_lines = [
                f"Pot: {stats.get('pot', 0):.1f} BB",
                f"To Call: {stats.get('to_call', 0):.1f} BB",
                f"Your Stack: {stats.get('my_stack', 0):.1f} BB",
                f"Opp Stack: {stats.get('opp_stack', 0):.1f} BB",
                f"Street: {stats.get('street', 'Unknown')}",
                f"Position: {stats.get('position', 'Unknown')}",
                f"SPR: {stats.get('spr', 0):.2f}",
            ]
            
            if stats.get('equity') is not None:
                stats_lines.append(f"Equity: {stats['equity']:.1f}%")
                stats_lines.append(f"Pot Odds: {stats.get('pot_odds', 0):.1f}%")
                stats_lines.append(f"Required: {stats.get('required_equity', 0):.1f}%")
                edge = stats['equity'] - stats.get('required_equity', 0)
                if edge > 0:
                    stats_lines.append(f"Edge: +{edge:.1f}%")
                else:
                    stats_lines.append(f"Deficit: {edge:.1f}%")
            
            if stats.get('rollout_evs'):
                stats_lines.append("\nRollout EVs:")
                for action_id, ev in sorted(stats['rollout_evs'].items(), key=lambda x: x[1], reverse=True):
                    action_name = self.action_buttons[action_id].cget("text") if action_id in self.action_buttons else f"Action {action_id}"
                    stats_lines.append(f"  {action_name}: {ev:+.2f} BB")
            
            self.stats_text.insert(1.0, "\n".join(stats_lines))
            
            # Update recommendation display
            self.rec_text.delete(1.0, tk.END)
            rec_lines = [
                f"Recommended: {rec.get('action_name', 'Unknown')}",
                "",
                rec.get('explanation', 'No explanation available.')
            ]
            self.rec_text.insert(1.0, "\n".join(rec_lines))
            
            # Update risk analysis
            self.risk_analysis_text.delete(1.0, tk.END)
            risk_analysis = rec.get('risk_analysis', {})
            if risk_analysis:
                risk_lines = []
                for risk_level in ['conservative', 'moderate', 'aggressive']:
                    if risk_level in risk_analysis:
                        analysis = risk_analysis[risk_level]
                        risk_lines.append(f"{risk_level.capitalize()}: {analysis.get('action_name', 'Unknown')}")
                        risk_lines.append(f"  {analysis.get('reasoning', '')}")
                        risk_lines.append("")
                self.risk_analysis_text.insert(1.0, "\n".join(risk_lines))
        except Exception as e:
            self.rec_text.delete(1.0, tk.END)
            self.rec_text.insert(1.0, f"Error getting recommendation: {e}")
    
    def new_hand(self):
        self.env.reset()
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(tk.END, "New Hand Started\n")
        self.update_display()
        self.update_recommendations()
        
        # If bot acts first
        if self.env.current_player == self.bot_idx:
            self.root.after(500, self.bot_action)

    def update_display(self):
        # Board
        board_str = " ".join([str(c) for c in self.env.board]) if self.env.board else "---"
        self.board_label.config(text=board_str)
        
        # Pot
        self.pot_label.config(text=f"{self.env.pot:.1f} BB")
        
        # Street
        streets = ["PREFLOP", "FLOP", "TURN", "RIVER"]
        self.street_label.config(text=streets[self.env.street])
        
        # Hands
        hero_hand = " ".join([str(c) for c in self.env.hands[self.human_idx]])
        self.player_hand_label.config(text=hero_hand)
        
        if self.env.done:
            bot_hand = " ".join([str(c) for c in self.env.hands[self.bot_idx]])
            self.bot_hand_label.config(text=bot_hand)
        else:
            self.bot_hand_label.config(text="?? ??")
            
        # Stacks
        self.player_stack_label.config(text=f"{self.env.stacks[self.human_idx]:.1f} BB")
        self.bot_stack_label.config(text=f"{self.env.stacks[self.bot_idx]:.1f} BB")
        
        # Roles
        btn_idx = self.env.button
        self.player_role_label.config(text="Button (IP)" if self.human_idx == btn_idx else "Big Blind (OOP)")
        self.bot_role_label.config(text="Button (IP)" if self.bot_idx == btn_idx else "Big Blind (OOP)")
        
        # Buttons
        if not self.env.done and self.env.current_player == self.human_idx:
            legal = self.env._get_legal_actions()
            for aid, btn in self.action_buttons.items():
                if aid in legal:
                    btn.state(["!disabled"])
                else:
                    btn.state(["disabled"])
            # Update recommendations when it's player's turn
            self.update_recommendations()
        else:
            for btn in self.action_buttons.values():
                btn.state(["disabled"])

    def human_action(self, action_id):
        if self.env.done or self.env.current_player != self.human_idx:
            return
            
        action_name = self.action_buttons[action_id].cget("text")
        self.history_text.insert(tk.END, f"You: {action_name}\n")
        self.history_text.see(tk.END)
        
        self.env.step(action_id)
        self.update_display()
        self.update_recommendations()
        
        if not self.env.done:
            self.root.after(500, self.bot_action)
        else:
            self.end_hand()

    def bot_action(self):
        if self.env.done or self.env.current_player != self.bot_idx:
            return
            
        # Get info set
        info_set = self.agent._get_info_set(self.env, self.bot_idx)
        legal_actions = self.env._get_legal_actions()
        
        # Get action from agent (uses average strategy)
        # Note: get_action samples from the strategy. 
        # For best play, we usually want the average strategy.
        # CFRAgent.get_action samples from get_strategy (current strategy).
        # We should use get_average_strategy and sample from that.
        
        avg_strat = self.agent.get_average_strategy(info_set, legal_actions)
        
        # Sample from average strategy
        # avg_strat is a numpy array of size num_actions (5 for abstracted)
        # np.random.choice needs the array size and probability distribution
        if len(avg_strat) == 0:
            # Fallback: use uniform random if empty
            abstract_action = np.random.randint(0, self.agent.num_actions)
        else:
            # Normalize probabilities to ensure they sum to 1
            prob_sum = np.sum(avg_strat)
            if prob_sum > 0:
                avg_strat = avg_strat / prob_sum
            else:
                # Uniform if all zeros
                avg_strat = np.ones(len(avg_strat)) / len(avg_strat)
            abstract_action = np.random.choice(len(avg_strat), p=avg_strat)
        
        # Map back
        # 0->0 (fold), 1->1 (call), 2->4 (1/3 pot), 3->6 (3/4 pot), 4->7 (pot)
        # Wait, the mapping in CFRAgent (line 88) is:
        # abstract_to_env = {0: 0, 1: 1, 2: 4, 3: 6, 4: 7}
        # This seems fixed.
        
        abstract_to_env = {0: 0, 1: 1, 2: 4, 3: 6, 4: 7}
        if abstract_action in abstract_to_env:
            env_action = abstract_to_env[abstract_action]
        else:
            env_action = abstract_action
            
        # Ensure it's legal in env. If not, fallback to Check/Call or Fold.
        if env_action not in legal_actions:
            # This can happen due to abstraction mismatch (e.g. bet size not exactly available)
            # Try to find closest legal action
            if env_action == 0 and 0 in legal_actions: pass
            elif env_action == 1 and 1 in legal_actions: pass
            else:
                # Fallback
                if 1 in legal_actions: env_action = 1
                else: env_action = 0
        
        # Log
        action_name = self.action_buttons[env_action].cget("text")
        self.history_text.insert(tk.END, f"Bot: {action_name}\n")
        self.history_text.see(tk.END)
        
        self.env.step(env_action)
        self.update_display()
        self.update_recommendations()
        
        if not self.env.done:
            # If it's still bot's turn (e.g. after raise?), loop
            if self.env.current_player == self.bot_idx:
                self.root.after(500, self.bot_action)
        else:
            self.end_hand()

    def end_hand(self):
        rewards = self.env.rewards
        if rewards[self.human_idx] > 0:
            res = f"You WON {rewards[self.human_idx]:.1f} BB!"
        elif rewards[self.human_idx] < 0:
            res = f"You LOST {abs(rewards[self.human_idx]):.1f} BB."
        else:
            res = "SPLIT POT."
            
        self.history_text.insert(tk.END, f"\n--- {res} ---\n")
        self.history_text.see(tk.END)
        self.update_display()
        self.update_recommendations()

def main():
    root = tk.Tk()
    # Path to your checkpoint - using the highest available checkpoint
    # You can change this to any checkpoint in checkpoints/cfr_abstracted/
    ckpt_path = os.path.join(os.path.dirname(__file__), '../checkpoints/cfr_abstracted/cfr_abstracted_200000.pkl')
    # Fallback to final checkpoint if 200k doesn't exist
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), '../checkpoints/cfr_abstracted/cfr_abstracted_final.pkl')
    app = CFRPokerGUI(root, checkpoint_path=ckpt_path)
    root.mainloop()

if __name__ == "__main__":
    main()

