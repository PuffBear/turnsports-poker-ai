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
        self.root.title("Heads-Up No-Limit Hold'em: You vs CFR Bot with AI Coach")
        self.root.geometry("1600x900")  # Wider for recommendation panel
        
        # Initialize environment
        self.env = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
        
        # Initialize abstractions
        self.card_abstraction = CardAbstraction()
        
        # Load K-Means models for proper abstraction
        kmeans_flop_path = 'data/kmeans/kmeans_flop_latest.pkl'
        kmeans_turn_path = 'data/kmeans/kmeans_turn_latest.pkl'
        if os.path.exists(kmeans_flop_path) and os.path.exists(kmeans_turn_path):
            print("Loading K-Means models for card abstraction...")
            self.card_abstraction.load_kmeans_models(kmeans_flop_path, kmeans_turn_path)
            print("✅ K-Means models loaded!")
        else:
            print("⚠️  K-Means models not found - using hash bucketing (bot will play poorly)")
        
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
        
        self.human_idx = 0
        self.bot_idx = 1
        
        # Session statistics
        self.session_stats = {
            'hands_played': 0,
            'player_total_bb': 0.0,
            'bot_total_bb': 0.0
        }
        
        # Initialize LLM Coach
        try:
            bot_policy = None  # Skip DQN policy loading
            self.llm_coach = LLMCoach(bot_policy=bot_policy, risk_tolerance='moderate')
            print("✅ LLM Coach initialized (equity/pot odds analysis)")
        except Exception as e:
            print(f"⚠️  Could not initialize LLM coach: {e}")
            print("Continuing without coach recommendations.")
            self.llm_coach = None
        
        # Setup UI
        self.setup_ui()
        
        # Start new hand
        self.new_hand()
