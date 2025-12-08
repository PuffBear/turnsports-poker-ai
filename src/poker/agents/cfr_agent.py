import numpy as np
from collections import defaultdict
import pickle

class CFRAgent:
    """
    Counterfactual Regret Minimization for Poker.
    
    This is THE algorithm for solving poker games.
    Converges to Nash equilibrium through self-play.
    """
    
    def __init__(self, card_abstraction=None, action_abstraction=None, num_actions=9):
        """
        Initialize CFR agent.
        
        Args:
            card_abstraction: Optional CardAbstraction instance for state bucketing
            action_abstraction: Optional ActionAbstraction instance for action simplification
            num_actions: Number of actions in the game (9 for full Hold'em, 5 for abstracted)
        """
        # Abstraction modules
        self.card_abstraction = card_abstraction
        self.action_abstraction = action_abstraction
        self.num_actions = num_actions
        
        # Strategy tables
        self.regret_sum = defaultdict(lambda: np.zeros(num_actions))  # Cumulative regrets
        self.strategy_sum = defaultdict(lambda: np.zeros(num_actions))  # Cumulative strategy
        self.iterations = 0
    
    def get_strategy(self, info_set, legal_actions):
        """
        Get current strategy for an information set using regret matching.
        
        Args:
            info_set: String representing the information set (cards + history)
            legal_actions: List of legal action indices
            
        Returns:
            strategy: Probability distribution over actions
        """
        regrets = self.regret_sum[info_set]
        strategy = np.zeros(self.num_actions)
        
        # Map legal actions if using action abstraction
        if self.action_abstraction:
            # Simple static mapping: 9 env actions -> 5 abstract actions
            # 0:fold->0, 1:check/call->1, 2-4:small bet->2, 5-8:large bet->3/4
            env_to_abstract = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}
            abstract_legal = set()
            for env_action in legal_actions:
                if env_action in env_to_abstract:
                    abstract_legal.add(env_to_abstract[env_action])
            legal_actions = list(abstract_legal)
        
        # Regret matching: positive regrets become probabilities
        positive_regrets = np.maximum(regrets, 0)
        
        # Only consider legal actions
        for i in range(self.num_actions):
            if i not in legal_actions:
                positive_regrets[i] = 0
        
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 0:
            # Normalize positive regrets
            strategy = positive_regrets / regret_sum
        else:
            # Uniform over legal actions
            for action in legal_actions:
                strategy[action] = 1.0 / len(legal_actions)
        
        return strategy
    
    def get_action(self, info_set, legal_actions):
        """Sample action from current strategy."""
        strategy = self.get_strategy(info_set, legal_actions)
        
        # Sample from strategy (returns abstract action if using abstraction)
        abstract_action = np.random.choice(self.num_actions, p=strategy)
        
        # Map back to environment action if using action abstraction
        if self.action_abstraction:
            # Simple reverse mapping: abstract -> env action
            # 0->0 (fold), 1->1 (call), 2->4 (1/3 pot), 3->6 (3/4 pot), 4->7 (pot)
            abstract_to_env = {0: 0, 1: 1, 2: 4, 3: 6, 4: 7}
            if abstract_action in abstract_to_env:
                return abstract_to_env[abstract_action]
        return abstract_action
    
    def get_average_strategy(self, info_set, legal_actions):
        """
        Get the average strategy (Nash approximation) for an info set.
        This is what we use for actual play after training.
        """
        avg_strategy = self.strategy_sum[info_set].copy()
        
        # Map legal actions if using action abstraction
        if self.action_abstraction:
            # Same static mapping as in get_strategy
            env_to_abstract = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}
            abstract_legal = set()
            for env_action in legal_actions:
                if env_action in env_to_abstract:
                    abstract_legal.add(env_to_abstract[env_action])
            legal_actions = list(abstract_legal)
        
        # Zero out illegal actions
        for i in range(self.num_actions):
            if i not in legal_actions:
                avg_strategy[i] = 0
        
        strategy_sum = np.sum(avg_strategy)
        
        if strategy_sum > 0:
            avg_strategy /= strategy_sum
        else:
            # Uniform over legal actions
            for action in legal_actions:
                avg_strategy[action] = 1.0 / len(legal_actions)
        
        return avg_strategy
    
    def update_strategy(self, info_set, strategy):
        """Add current strategy to cumulative strategy."""
        self.strategy_sum[info_set] += strategy
    
    def update_regrets(self, info_set, action_regrets):
        """Update cumulative regrets for an information set."""
        self.regret_sum[info_set] += action_regrets
    
    def train_iteration(self, env, player_idx):
        """
        Run one CFR iteration from the perspective of player_idx using External Sampling.
        
        External Sampling CFR samples one action per node instead of exploring all,
        making it much more efficient for large game trees.
        
        Returns:
            expected_value: The value of the game for player_idx
        """
        # Reset environment
        state, info = env.reset()
        
        # Traverse game tree with external sampling
        return self._external_sampling_cfr(env, player_idx)
    
    def _external_sampling_cfr(self, env, player_idx):
        """
        External Sampling CFR - samples actions instead of exploring all branches.
        Much more efficient than vanilla CFR for poker.
        
        Args:
            env: Game environment (will be modified in place)
            player_idx: Player we're computing values for (0 or 1)
            
        Returns:
            utility: Utility at this node for player_idx
        """
        # Check if game is over
        if env.done:
            # Return the actual payoff
            # The environment stores the last reward
            # Need to get it for the correct player
            if hasattr(env, 'rewards') and env.rewards is not None:
                return env.rewards[player_idx]
            return 0.0
        
        current_player = env.current_player
        legal_actions = env._get_legal_actions()
        
        # If no legal actions, game should be done
        if len(legal_actions) == 0:
            return 0.0
        
        # Create information set string
        info_set = self._get_info_set(env, current_player)
        
        # Get current strategy
        strategy = self.get_strategy(info_set, legal_actions)
        
        if current_player == player_idx:
            # Player we're updating: compute counterfactual values
            
            # We need to try all actions to compute regrets
            # Save environment state
            saved_state = self._save_env_state(env)
            
            action_utilities = np.zeros(self.num_actions)
            
            # Get abstract legal actions if using abstraction
            if self.action_abstraction:
                env_to_abstract = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}
                abstract_to_env = {0: 0, 1: 1, 2: 4, 3: 6, 4: 7}
                abstract_legal = set()
                for env_action in legal_actions:
                    if env_action in env_to_abstract:
                        abstract_legal.add(env_to_abstract[env_action])
                actions_to_try = list(abstract_legal)
            else:
                actions_to_try = legal_actions
                abstract_to_env = None
            
            for abstract_action in actions_to_try:
                # Restore state
                self._restore_env_state(env, saved_state)
                
                # Map to env action if needed
                if self.action_abstraction and abstract_to_env:
                    env_action = abstract_to_env.get(abstract_action, abstract_action)
                else:
                    env_action = abstract_action
                
                # Take action
                next_state, reward, done, truncated, info_dict = env.step(env_action)
                
                # Recurse
                action_utilities[abstract_action] = self._external_sampling_cfr(env, player_idx)
            
            # Expected utility
            utility = np.sum(strategy * action_utilities)
            
            # Update regrets
            regrets = action_utilities - utility
            self.update_regrets(info_set, regrets)
            
            # Update strategy sum (weighted by reach probability, which is 1 in external sampling)
            self.update_strategy(info_set, strategy)
            
            # Restore state for caller
            self._restore_env_state(env, saved_state)
            
            return utility
        else:
            # Opponent: sample according to strategy
            sampled_action = np.random.choice(self.num_actions, p=strategy)
            
            # Map to environment action if using abstraction
            if self.action_abstraction:
                abstract_to_env = {0: 0, 1: 1, 2: 4, 3: 6, 4: 7}
                if sampled_action in abstract_to_env:
                    env_action = abstract_to_env[sampled_action]
                else:
                    env_action = sampled_action
            else:
                env_action = sampled_action
            
            # Take sampled action
            next_state, reward, done, truncated, info_dict = env.step(env_action)
            
            # Recurse
            return self._external_sampling_cfr(env, player_idx)
    
    def _save_env_state(self, env):
        """Save minimal environment state needed for CFR."""
        # Optimization: Use list slicing/comprehension instead of deepcopy
        # Cards are immutable, so we only need to copy the lists
        return {
            'hands': [list(h) for h in env.hands],
            'board': list(env.board),
            'deck_cards': list(env.deck.cards),
            'pot': env.pot,
            'stacks': list(env.stacks),
            'street_investment': list(env.street_investment),
            'street': env.street,
            'current_player': env.current_player,
            'done': env.done,
            'has_acted': list(env.has_acted)
        }
    
    def _restore_env_state(self, env, state):
        """Restore environment state."""
        env.hands = [list(h) for h in state['hands']]
        env.board = list(state['board'])
        env.deck.cards = list(state['deck_cards'])
        env.pot = state['pot']
        env.stacks = list(state['stacks'])
        env.street_investment = list(state['street_investment'])
        env.street = state['street']
        env.current_player = state['current_player']
        env.done = state['done']
        env.has_acted = list(state['has_acted'])
    
    def _get_info_set(self, env, player):
        """
        Create information set string from game state.
        Only includes information visible to the player.
        
        If card_abstraction is enabled, uses buckets instead of raw cards.
        """
        # Determine street name
        street_names = ['preflop', 'flop', 'turn', 'river']
        street_name = street_names[env.street] if env.street < len(street_names) else 'river'
        
        if self.card_abstraction:
            # Use abstraction: bucket instead of raw cards
            hand = env.hands[player]
            board = env.board
            
            # Convert Card objects to strings
            hand_strs = [str(c) for c in hand]
            board_strs = [str(c) for c in board]
            
            # Get abstract bucket
            bucket = self.card_abstraction.get_bucket(hand_strs, board_strs, street_name)
            
            # Simplified info set with bucket
            pot = int(env.pot)
            to_call = int(abs(env.street_investment[0] - env.street_investment[1]))
            
            info_set = f"{bucket}|{street_name}|{pot}|{to_call}"
        else:
            # Original: use raw card strings
            hand = env.hands[player]
            hand_str = ''.join(sorted([str(c) for c in hand]))
            
            # Get board
            board_str = ''.join([str(c) for c in env.board])
            
            # Get betting history (simplified)
            street = env.street
            pot = int(env.pot)
            to_call = int(abs(env.street_investment[0] - env.street_investment[1]))
            
            # Create info set string
            info_set = f"{hand_str}|{board_str}|{street}|{pot}|{to_call}"
        
        return info_set
    
    def save(self, path):
        """Save CFR tables."""
        data = {
            'regret_sum': dict(self.regret_sum),
            'strategy_sum': dict(self.strategy_sum),
            'iterations': self.iterations
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path):
        """Load CFR tables."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.regret_sum = defaultdict(lambda: np.zeros(self.num_actions), data['regret_sum'])
        self.strategy_sum = defaultdict(lambda: np.zeros(self.num_actions), data['strategy_sum'])
        self.iterations = data['iterations']
