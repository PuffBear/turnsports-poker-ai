"""
RL-CFR Agent for Kuhn Poker.

This is a reference implementation combining:
- Reinforcement Learning (neural network function approximation)
- Counterfactual Regret Minimization (game theory)

Kuhn Poker is a toy game used to demonstrate the concepts before
scaling to Texas Hold'em.

Key ideas:
1. Instead of storing regrets in tables, use neural networks
2. Networks generalize to unseen states
3. Suitable for larger games where tabular CFR doesn't fit in memory

References:
- "Deep Counterfactual Regret Minimization" (Brown et al., 2019)
- "Solving Imperfect Information Games Using Reinforcement Learning" 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class RegretNetwork(nn.Module):
    """
    Neural network that predicts regrets for each action.
    
    Input: State representation
    Output: Regret values for each action
    """
    
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        super(RegretNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, state):
        """Forward pass: state -> regrets."""
        return self.network(state)


class AverageStrategyNetwork(nn.Module):
    """
    Neural network that outputs action probabilities.
    
    This approximates the average strategy (Nash approximation).
    
    Input: State representation
    Output: Probability distribution over actions
    """
    
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        super(AverageStrategyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """Forward pass: state -> action probabilities."""
        return self.network(state)


class RLCFRAgent:
    """
    RL-CFR Agent for Kuhn Poker.
    
    Combines CFR algorithm with neural network function approximation.
    
    Usage:
        agent = RLCFRAgent(state_dim=6, num_actions=2)
        
        for iteration in range(10000):
            agent.train_iteration(env, player_idx=0)
            agent.train_iteration(env, player_idx=1)
        
        # Get action
        state = env.get_state_vector()
        strategy = agent.get_strategy(state, legal_actions)
    """
    
    def __init__(self, state_dim: int = 6, num_actions: int = 2, 
                 hidden_dim: int = 64, learning_rate: float = 1e-3):
        """
        Initialize RL-CFR agent.
        
        Args:
            state_dim: Dimension of state vector
            num_actions: Number of possible actions
            hidden_dim: Hidden layer size for networks
            learning_rate: Learning rate for gradient descent
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Neural networks
        self.regret_net = RegretNetwork(state_dim, num_actions, hidden_dim)
        self.avg_strategy_net = AverageStrategyNetwork(state_dim, num_actions, hidden_dim)
        
        # Optimizers
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.avg_strategy_optimizer = optim.Adam(self.avg_strategy_net.parameters(), lr=learning_rate)
        
        # Memory buffers for training
        self.regret_memory = deque(maxlen=100000)  # (state, regrets)
        self.strategy_memory = deque(maxlen=100000)  # (state, strategy)
        
        # Training config
        self.batch_size = 256
        self.update_frequency = 100  # Update networks every N iterations
        self.iteration = 0
    
    def get_strategy(self, state: np.ndarray, legal_actions: list) -> np.ndarray:
        """
        Get current strategy using regret matching.
        
        Args:
            state: State vector (numpy array)
            legal_actions: List of legal action indices
        
        Returns:
            strategy: Probability distribution over actions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            regrets = self.regret_net(state_tensor).squeeze(0).numpy()
        
        # Regret matching
        strategy = np.zeros(self.num_actions)
        positive_regrets = np.maximum(regrets, 0)
        
        # Only consider legal actions
        for i in range(self.num_actions):
            if i not in legal_actions:
                positive_regrets[i] = 0
        
        regret_sum = positive_regrets.sum()
        
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            # Uniform over legal actions
            for action in legal_actions:
                strategy[action] = 1.0 / len(legal_actions)
        
        return strategy
    
    def get_average_strategy(self, state: np.ndarray, legal_actions: list) -> np.ndarray:
        """
        Get average strategy (Nash approximation) from network.
        
        This is used for actual play after training.
        
        Args:
            state: State vector
            legal_actions: List of legal action indices
        
        Returns:
            avg_strategy: Probability distribution over actions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            strategy = self.avg_strategy_net(state_tensor).squeeze(0).numpy()
        
        # Mask illegal actions
        for i in range(self.num_actions):
            if i not in legal_actions:
                strategy[i] = 0
        
        # Renormalize
        strategy_sum = strategy.sum()
        if strategy_sum > 0:
            strategy /= strategy_sum
        else:
            # Fallback: uniform over legal actions
            for action in legal_actions:
                strategy[action] = 1.0 / len(legal_actions)
        
        return strategy
    
    def train_iteration(self, env, player_idx: int):
        """
        Run one CFR iteration from the perspective of player_idx.
        
        This is the core RL-CFR algorithm:
        1. Traverse game tree
        2. Compute counterfactual values
        3. Calculate regrets
        4. Store (state, regret) and (state, strategy) pairs
        5. Periodically update networks
        
        Args:
            env: Kuhn Poker environment
            player_idx: Player to update (0 or 1)
        
        Returns:
            utility: Expected utility for player_idx
        """
        # Reset environment
        env.reset()
        
        # Traverse game tree
        utility = self._cfr_traverse(env, player_idx, reach_probs=[1.0, 1.0])
        
        # Update networks periodically
        self.iteration += 1
        if self.iteration % self.update_frequency == 0:
            self._update_networks()
        
        return utility
    
    def _cfr_traverse(self, env, player_idx: int, reach_probs: list):
        """
        CFR traversal with neural network function approximation.
        
        Args:
            env: Environment (will be modified in-place)
            player_idx: Player we're computing values for
            reach_probs: Reach probabilities for both players
        
        Returns:
            utility: Utility at this node for player_idx
        """
        # Check if terminal
        if env.done:
            # Return payoff for player_idx
            if hasattr(env, 'rewards') and env.rewards is not None:
                return env.rewards[player_idx]
            return 0.0
        
        current_player = env.current_player
        legal_actions = env.get_legal_actions()
        
        if len(legal_actions) == 0:
            return 0.0
        
        # Get state representation
        state = env.get_state_vector(player_idx=current_player)
        
        # Get current strategy
        strategy = self.get_strategy(state, legal_actions)
        
        if current_player == player_idx:
            # Player we're updating: compute counterfactual values
            
            # Save environment state (simple for Kuhn poker)
            saved_state = self._save_state(env)
            
            action_utilities = np.zeros(self.num_actions)
            
            for action in legal_actions:
                # Restore state
                self._restore_state(env, saved_state)
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                
                # Recurse
                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] *= strategy[action]
                
                action_utilities[action] = self._cfr_traverse(env, player_idx, new_reach_probs)
            
            # Expected utility
            utility = np.sum(strategy * action_utilities)
            
            # Compute regrets
            regrets = action_utilities - utility
            
            # Weight by opponent reach probability (standard CFR)
            opponent_reach = reach_probs[1 - current_player]
            weighted_regrets = regrets * opponent_reach
            
            # Store training data
            self.regret_memory.append((state, weighted_regrets))
            self.strategy_memory.append((state, strategy))
            
            # Restore state for caller
            self._restore_state(env, saved_state)
            
            return utility
        
        else:
            # Opponent: sample according to strategy (external sampling)
            sampled_action = np.random.choice(self.num_actions, p=strategy)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(sampled_action)
            
            # Recurse
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[sampled_action]
            
            return self._cfr_traverse(env, player_idx, new_reach_probs)
    
    def _save_state(self, env):
        """Save environment state (simplified for Kuhn poker)."""
        return {
            'cards': env.cards.copy() if hasattr(env, 'cards') else None,
            'pot': env.pot if hasattr(env, 'pot') else 0,
            'current_player': env.current_player if hasattr(env, 'current_player') else 0,
            'done': env.done if hasattr(env, 'done') else False,
            'history': env.history.copy() if hasattr(env, 'history') else [],
        }
    
    def _restore_state(self, env, state):
        """Restore environment state."""
        if state['cards'] is not None:
            env.cards = state['cards'].copy()
        if hasattr(env, 'pot'):
            env.pot = state['pot']
        if hasattr(env, 'current_player'):
            env.current_player = state['current_player']
        if hasattr(env, 'done'):
            env.done = state['done']
        if hasattr(env, 'history'):
            env.history = state['history'].copy()
    
    def _update_networks(self):
        """
        Update neural networks using stored training data.
        
        Uses supervised learning:
        - Regret network: MSE loss to fit accumulated regrets
        - Strategy network: Cross-entropy loss to fit average strategy
        """
        if len(self.regret_memory) < self.batch_size:
            return
        
        # Sample mini-batch
        regret_batch = random.sample(self.regret_memory, min(self.batch_size, len(self.regret_memory)))
        strategy_batch = random.sample(self.strategy_memory, min(self.batch_size, len(self.strategy_memory)))
        
        # Prepare data
        regret_states = torch.FloatTensor([x[0] for x in regret_batch])
        regret_targets = torch.FloatTensor([x[1] for x in regret_batch])
        
        strategy_states = torch.FloatTensor([x[0] for x in strategy_batch])
        strategy_targets = torch.FloatTensor([x[1] for x in strategy_batch])
        
        # Update regret network
        self.regret_optimizer.zero_grad()
        regret_preds = self.regret_net(regret_states)
        regret_loss = nn.MSELoss()(regret_preds, regret_targets)
        regret_loss.backward()
        self.regret_optimizer.step()
        
        # Update average strategy network
        self.avg_strategy_optimizer.zero_grad()
        strategy_preds = self.avg_strategy_net(strategy_states)
        
        # KL divergence loss
        strategy_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(strategy_preds + 1e-8), 
            strategy_targets
        )
        strategy_loss.backward()
        self.avg_strategy_optimizer.step()
        
        if self.iteration % 1000 == 0:
            print(f"[Iteration {self.iteration}] Regret Loss: {regret_loss.item():.4f}, Strategy Loss: {strategy_loss.item():.4f}")
    
    def save(self, path: str):
        """Save agent networks and state."""
        torch.save({
            'regret_net': self.regret_net.state_dict(),
            'avg_strategy_net': self.avg_strategy_net.state_dict(),
            'regret_optimizer': self.regret_optimizer.state_dict(),
            'avg_strategy_optimizer': self.avg_strategy_optimizer.state_dict(),
            'iteration': self.iteration,
        }, path)
        print(f"✅ Saved RL-CFR agent to {path}")
    
    def load(self, path: str):
        """Load agent networks and state."""
        checkpoint = torch.load(path)
        self.regret_net.load_state_dict(checkpoint['regret_net'])
        self.avg_strategy_net.load_state_dict(checkpoint['avg_strategy_net'])
        self.regret_optimizer.load_state_dict(checkpoint['regret_optimizer'])
        self.avg_strategy_optimizer.load_state_dict(checkpoint['avg_strategy_optimizer'])
        self.iteration = checkpoint['iteration']
        print(f"✅ Loaded RL-CFR agent from {path}")


if __name__ == "__main__":
    print("RL-CFR Agent for Kuhn Poker")
    print("=" * 60)
    print("\nThis module provides a neural network-based CFR agent.")
    print("To train, use: experiments/kuhn/train_kuhn_rlcfr.py")
    print()
