import copy
import numpy as np

def rollout_action(env, action_id, bot_policy, n_episodes=200):
    """
    Simulate taking an action and playing out the hand.
    
    Args:
        env: Current environment state (will be cloned)
        action_id: Action to take
        bot_policy: Bot's policy wrapper
        n_episodes: Number of rollouts
    
    Returns:
        avg_ev: Average EV in BB (big blinds)
    """
    player = env.current_player
    total_reward = 0
    
    for _ in range(n_episodes):
        # Clone environment
        env_copy = copy.deepcopy(env)
        
        # Take the specified action
        obs, reward, done, truncated, info = env_copy.step(action_id)
        total_reward += reward
        
        if done:
            continue
        
        # Play out rest of hand with bot policy
        max_steps = 100  # Prevent infinite loops
        steps = 0
        
        while not done and steps < max_steps:
            # Get action from bot
            legal_actions = env_copy._get_legal_actions()
            bot_action = bot_policy.get_action(obs, legal_actions, deterministic=True)
            
            obs, reward, done, truncated, info = env_copy.step(bot_action)
            
            # Accumulate reward if this player
            if env_copy.current_player == player or done:
                total_reward += reward
            
            steps += 1
    
    avg_ev = total_reward / n_episodes
    return avg_ev

def compare_actions(env, candidate_actions, bot_policy, n_episodes=200):
    """
    Compare EV of multiple candidate actions.
    
    Args:
        env: Current environment state
        candidate_actions: List of action IDs to compare
        bot_policy: Bot's policy wrapper
        n_episodes: Number of rollouts per action
    
    Returns:
        dict: {action_id: avg_ev}
    """
    results = {}
    
    for action_id in candidate_actions:
        ev = rollout_action(env, action_id, bot_policy, n_episodes)
        results[action_id] = ev
    
    return results
