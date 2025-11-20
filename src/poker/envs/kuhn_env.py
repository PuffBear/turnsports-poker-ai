import gymnasium as gym
import numpy as np
from gymnasium import spaces

class KuhnPokerEnv(gym.Env):
    """
    Kuhn Poker Environment.
    Cards: 0 (J), 1 (Q), 2 (K)
    Actions: 0 (Check/Fold), 1 (Bet/Call)
    """
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3) # Card rank
        self.cards = [0, 1, 2]
        self.reset()

    def reset(self):
        np.random.shuffle(self.cards)
        self.p1_card = self.cards[0]
        self.p2_card = self.cards[1]
        self.current_player = 0 # 0 for P1, 1 for P2
        self.history = [] # List of actions taken
        return self._get_obs()

    def _get_obs(self):
        # Simple observation: just the card rank
        # In a real RL setting, we'd also include history
        card = self.p1_card if self.current_player == 0 else self.p2_card
        return card

    def step(self, action):
        # action: 0 = PASS (Check/Fold), 1 = BET (Bet/Call)
        self.history.append(action)
        
        reward = 0
        done = False
        
        # Terminal states
        # P1 Check, P2 Check -> Showdown (1)
        if self.history == [0, 0]:
            reward = self._showdown(1)
            done = True
        
        # P1 Bet, P2 Fold -> P1 wins (1)
        elif self.history == [1, 0]:
            reward = 1 if self.current_player == 0 else -1 # Current player is P2 who folded? No, wait.
            # If history is [1, 0]:
            # P1 Bet (1)
            # P2 Fold (0) -> P1 wins 1.
            # If we are calling step(0) for P2:
            # We return reward for P2. P2 folded, so P2 loses 1.
            reward = -1
            done = True
            
        # P1 Bet, P2 Call -> Showdown (2)
        elif self.history == [1, 1]:
            reward = self._showdown(2)
            done = True
            
        # P1 Check, P2 Bet, P1 Fold -> P2 wins (1)
        elif self.history == [0, 1, 0]:
            # P1 just folded. P1 loses 1.
            reward = -1
            done = True
            
        # P1 Check, P2 Bet, P1 Call -> Showdown (2)
        elif self.history == [0, 1, 1]:
            reward = self._showdown(2)
            done = True
            
        if not done:
            self.current_player = 1 - self.current_player
            
        return self._get_obs(), reward, done, {"history": self.history}

    def _showdown(self, pot_contribution):
        # Returns reward for the player who just acted (or rather, relative to the player who just acted?)
        # Usually in self-play we return reward for the player whose turn it was.
        # But standard gym is single agent. Let's assume we return reward for the agent.
        # Actually for CFR/Self-play we usually need the payoff for the terminal node.
        
        # Let's define reward as: payoff for P1.
        # Wait, step() returns reward for the agent taking the action?
        # If I am P1 and I act, I get a reward? Usually 0 until terminal.
        
        # Let's stick to: Return reward for the player who just made the move if terminal.
        # If P2 calls (1,1), P2 is current. P2 wins or loses.
        
        winner = 0
        if self.p1_card > self.p2_card:
            winner = 0
        else:
            winner = 1
            
        # If winner is current player, reward +pot. Else -pot.
        # pot_contribution is what EACH player put in (1 ante + bets).
        # Total pot is 2 * (1 + bets).
        # Profit is pot_contribution + bets.
        
        # Example: [1, 1]. Ante 1. Bet 1. Total put in: 2 each.
        # Winner gets 4 total. Profit +2. Loser -2.
        
        payoff = pot_contribution
        
        if winner == self.current_player:
            return payoff
        else:
            return -payoff
