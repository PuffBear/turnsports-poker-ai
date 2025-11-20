class PolicyWrapper:
    """
    Wrapper interface for bot policy to use in the coach/GUI.
    """
    def __init__(self, agent):
        self.agent = agent
    
    def get_action_probs(self, state, legal_actions):
        """
        Get action probabilities for display in GUI.
        Returns dict: {action_id: probability}
        """
        q_values = self.agent.get_q_values(state)
        
        # Softmax over legal actions
        import numpy as np
        legal_q = np.array([q_values[a] for a in legal_actions])
        
        # Shift for numerical stability
        legal_q = legal_q - np.max(legal_q)
        exp_q = np.exp(legal_q)
        probs = exp_q / np.sum(exp_q)
        
        return {a: p for a, p in zip(legal_actions, probs)}
    
    def get_action(self, state, legal_actions, deterministic=True):
        """
        Get action from policy.
        If deterministic, use greedy action. Otherwise use epsilon-greedy.
        """
        if deterministic:
            return self.agent.select_action(state, legal_actions, epsilon=0.0)
        else:
            return self.agent.select_action(state, legal_actions)
