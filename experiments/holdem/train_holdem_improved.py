import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from tqdm import tqdm
from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.agents.holdem_dqn import DQNAgent
from src.poker.opponents.holdem_rule_based import RandomOpponent, TAGOpponent, LAGOpponent

def train_dqn_improved(n_episodes=100000, save_interval=10000, device='cpu'):
    """
    Train DQN agent with improved hyperparameters and training strategy.
    """
    print("=" * 60)
    print("Training IMPROVED DQN Agent for Heads-Up NLHE")
    print("=" * 60)
    
    # Initialize environment
    env = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
    
    # Initialize agent with BETTER hyperparameters
    state_dim = 200
    action_dim = 9
    agent = DQNAgent(
        state_dim, 
        action_dim, 
        lr=3e-4,  # Higher learning rate for faster learning
        gamma=0.99, 
        epsilon_start=1.0, 
        epsilon_end=0.05,  # Lower final epsilon for more exploitation
        epsilon_decay=0.99995,  # Slower decay
        device=device
    )
    
    # Opponent pool with CURRICULUM LEARNING
    all_opponents = {
        'random': RandomOpponent(),
        'tag': TAGOpponent(),
        'lag': LAGOpponent()
    }
    
    # Training stats
    episode_rewards = []
    episode_lengths = []
    losses = []
    win_rates = {'random': [], 'tag': [], 'lag': []}
    
    print(f"\nTraining for {n_episodes} episodes...")
    print(f"Improved hyperparameters:")
    print(f"  Learning rate: 3e-4 (higher)")
    print(f"  Final epsilon: 0.05 (lower)")
    print(f"  Curriculum learning: Start easy → hard")
    print(f"Device: {device}\n")
    
    for episode in tqdm(range(n_episodes)):
        # CURRICULUM: Start with random, gradually add harder opponents
        if episode < 20000:
            opponent_name = 'random'
        elif episode < 60000:
            opponent_name = np.random.choice(['random', 'tag'], p=[0.5, 0.5])
        else:
            opponent_name = np.random.choice(['random', 'tag', 'lag'], p=[0.3, 0.4, 0.3])
        
        opponent = all_opponents[opponent_name]
        
        # Reset environment
        state, info = env.reset()
        agent_player = 0
        
        # Track transitions
        transitions = []
        done = False
        
        while not done:
            current_player = env.current_player
            legal_actions = env._get_legal_actions()
            
            if current_player == agent_player:
                action = agent.select_action(state, legal_actions)
                prev_state = state
                prev_action = action
            else:
                action = opponent.get_action(env, state)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            if current_player == agent_player:
                transitions.append((prev_state, prev_action, reward, next_state, done))
            
            state = next_state
        
        # Calculate total reward
        total_reward = sum(r for (_, _, r, _, _) in transitions)
        
        # Store transitions
        for i, (s, a, r, ns, d) in enumerate(transitions):
            if i == len(transitions) - 1:
                agent.store_transition(s, a, total_reward, ns, d)
            else:
                agent.store_transition(s, a, 0, ns, False)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(len(transitions))
        
        # Track win against this opponent type
        if total_reward > 0:
            win_rates[opponent_name].append(1)
        else:
            win_rates[opponent_name].append(0)
        
        # Train MORE aggressively
        if len(agent.replay_buffer) > 1000:
            # Train MULTIPLE times per episode for faster learning
            for _ in range(3):
                loss = agent.train_step(batch_size=128)  # Larger batches
                if loss is not None:
                    losses.append(loss)
        
        # Update target network more frequently
        if episode % 50 == 0:
            agent.update_target_network()
        
        # Save checkpoint and evaluate
        if (episode + 1) % save_interval == 0:
            checkpoint_path = f'checkpoints/improved_dqn_episode_{episode+1}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(checkpoint_path)
            
            # Print detailed stats
            avg_reward = np.mean(episode_rewards[-1000:])
            avg_length = np.mean(episode_lengths[-1000:])
            avg_loss = np.mean(losses[-1000:]) if losses else 0
            
            print(f"\n[Episode {episode+1}]")
            print(f"  Avg Reward (last 1000): {avg_reward:.2f} BB")
            print(f"  Avg Episode Length: {avg_length:.1f} steps")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            
            # Win rates
            for opp_name in ['random', 'tag', 'lag']:
                if win_rates[opp_name]:
                    wr = np.mean(win_rates[opp_name][-500:]) * 100
                    print(f"  Win rate vs {opp_name.upper()}: {wr:.1f}%")
    
    # Final save
    agent.save('checkpoints/improved_dqn_final.pt')
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to checkpoints/improved_dqn_final.pt")
    print("=" * 60)
    
    return agent

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = train_dqn_improved(n_episodes=100000, save_interval=10000, device=device)
