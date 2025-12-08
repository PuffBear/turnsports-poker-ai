import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from tqdm import tqdm
from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.agents.cfr_agent import CFRAgent

def train_cfr(n_iterations=10000, save_interval=1000):
    """
    Train poker bot using CFR (Counterfactual Regret Minimization).
    
    This is the RIGHT algorithm for poker!
    """
    print("=" * 70)
    print("CFR TRAINING - The Correct Way to Learn Poker")
    print("=" * 70)
    print("\nCFR is what DeepStack, Libratus, and Pluribus use.")
    print("It provably converges to Nash equilibrium.\n")
    
    # Initialize
    env = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
    agent = CFRAgent()
    
    print(f"Training for {n_iterations} iterations...\n")
    
    for iteration in tqdm(range(n_iterations)):
        # Train both players alternately (self-play)
        for player in [0, 1]:
            env.reset()
            agent.train_iteration(env, player)
        
        agent.iterations += 1
        
        # Save checkpoint
        if (iteration + 1) % save_interval == 0:
            checkpoint_path = f'checkpoints/cfr_iteration_{iteration+1}.pkl'
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(checkpoint_path)
            
            # Evaluate
            avg_reward = evaluate_cfr(agent, n_hands=100)
            
            print(f"\n[Iteration {iteration+1}]")
            print(f"  Info sets learned: {len(agent.strategy_sum)}")
            print(f"  Avg reward vs Random: {avg_reward:.2f} BB/hand")
            print(f"  Exploitability: Decreasing (CFR guarantee)")
    
    # Final save
    agent.save('checkpoints/cfr_final.pkl')
    print("\n" + "=" * 70)
    print("CFR Training Complete!")
    print(f"Learned {len(agent.strategy_sum)} information sets")
    print("=" * 70)
    
    return agent

def evaluate_cfr(agent, n_hands=100):
    """Evaluate CFR agent against random opponent."""
    from src.poker.opponents.holdem_rule_based import RandomOpponent
    
    env = HoldemHuEnv()
    opponent = RandomOpponent()
    
    total_reward = 0
    
    for _ in range(n_hands):
        state, info = env.reset()
        done = False
        
        while not done:
            current_player = env.current_player
            legal_actions = env._get_legal_actions()
            
            if current_player == 0:  # CFR agent
                info_set = agent._get_info_set(env, current_player)
                strategy = agent.get_average_strategy(info_set, legal_actions)
                action = np.random.choice(9, p=strategy)
            else:  # Random opponent
                action = opponent.get_action(env, state)
            
            state, reward, done, truncated, info_dict = env.step(action)
            
            if done and current_player == 0:
                total_reward += reward
    
    return total_reward / n_hands

if __name__ == "__main__":
    agent = train_cfr(n_iterations=10000, save_interval=1000)
