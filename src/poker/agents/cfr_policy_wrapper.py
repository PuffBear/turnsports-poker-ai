"""
Wrapper for CFR agent to use with the coach's rollout tools.
"""
import numpy as np

class CFRPolicyWrapper:
    """Wrapper for CFR agent that provides a policy interface for rollouts."""
    
    def __init__(self, cfr_agent):
        self.agent = cfr_agent
    
    def get_action(self, env, legal_actions, deterministic=True):
        """Get action from CFR agent's average strategy."""
        current_player = env.current_player
        info_set = self.agent._get_info_set(env, current_player)
        avg_strategy = self.agent.get_average_strategy(info_set, legal_actions)
        
        if deterministic:
            abstract_action = np.argmax(avg_strategy)
        else:
            prob_sum = np.sum(avg_strategy)
            if prob_sum > 0:
                avg_strategy = avg_strategy / prob_sum
            else:
                avg_strategy = np.ones(len(avg_strategy)) / len(avg_strategy)
            abstract_action = np.random.choice(len(avg_strategy), p=avg_strategy)
        
        # Map abstract action to environment action
        abstract_to_env = {0: 0, 1: 1, 2: 4, 3: 6, 4: 7}
        env_action = abstract_to_env.get(abstract_action, abstract_action)
        
        # Ensure it's legal
        if env_action not in legal_actions:
            env_action = 1 if 1 in legal_actions else legal_actions[0]
        
        return env_action
