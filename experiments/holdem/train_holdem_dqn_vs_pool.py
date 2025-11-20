import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from tqdm import tqdm
from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.agents.holdem_dqn import DQNAgent
from src.poker.opponents.holdem_rule_based import RandomOpponent, TAGOpponent, LAGOpponent

def train_dqn_vs_pool(n_episodes=50000, save_interval=5000, device='cpu'):
    """
    Train DQN agent against a pool of opponents.
    """
    print("=" * 60)
    print("Training DQN Agent for Heads-Up No-Limit Hold'em")
    print("=" * 60)
    
    # Initialize environment
    env = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
    
    # Initialize agent
    state_dim = 200  # Match observation space
    action_dim = 9   # 9 discrete actions
    agent = DQNAgent(state_dim, action_dim, lr=1e-4, gamma=0.99, 
                     epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9995,
                     device=device)
    
    # Opponent pool
    opponents = [
        RandomOpponent(),
        TAGOpponent(),
        LAGOpponent()
    ]
    
    # Training stats
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    print(f"\nTraining for {n_episodes} episodes...")
    print(f"Opponent pool: Random, TAG, LAG")
    print(f"Device: {device}")
    print()
    
    for episode in tqdm(range(n_episodes)):
        # Reset environment
        state, info = env.reset()
        
        # Select opponent
        opponent = np.random.choice(opponents)
        
        # Agent is always player 0
        agent_player = 0
        
        # Track transitions for this episode
        transitions = []
        done = False
        
        while not done:
            current_player = env.current_player
            legal_actions = env._get_legal_actions()
            
            if current_player == agent_player:
                # Agent's turn - store state/action for later
                action = agent.select_action(state, legal_actions)
                prev_state = state
                prev_action = action
            else:
                # Opponent's turn
                action = opponent.get_action(env, state)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # If agent just acted, save the transition
            if current_player == agent_player:
                transitions.append((prev_state, prev_action, reward, next_state, done))
            
            state = next_state
        
        #  Calculate final reward for completed hand
        # Sum all rewards the agent received
        total_reward = sum(r for (_, _, r, _, _) in transitions)
        
        # Store all transitions with proper rewards
        for i, (s, a, r, ns, d) in enumerate(transitions):
            # Only the final transition gets the actual reward
            if i == len(transitions) - 1:
                agent.store_transition(s, a, total_reward, ns, d)
            else:
                agent.store_transition(s, a, 0, ns, False)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(len(transitions))
        
        # Train agent
        if len(agent.replay_buffer) > 1000:
            loss = agent.train_step(batch_size=64)
            if loss is not None:
                losses.append(loss)
        
        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = f'checkpoints/dqn_episode_{episode+1}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(checkpoint_path)
            
            # Print stats
            avg_reward = np.mean(episode_rewards[-1000:])
            avg_length = np.mean(episode_lengths[-1000:])
            avg_loss = np.mean(losses[-1000:]) if losses else 0
            
            print(f"\n[Episode {episode+1}]")
            print(f"  Avg Reward (last 1000): {avg_reward:.2f} BB")
            print(f"  Avg Episode Length: {avg_length:.1f} steps")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Replay Buffer: {len(agent.replay_buffer)}")
    
    # Final save
    agent.save('checkpoints/dqn_final.pt')
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to checkpoints/dqn_final.pt")
    print("=" * 60)
    
    return agent

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = train_dqn_vs_pool(n_episodes=50000, save_interval=5000, device=device)
