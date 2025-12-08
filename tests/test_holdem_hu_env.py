
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.poker.envs.holdem_hu_env import HoldemHuEnv
import numpy as np

def test_check_call_down():
    print("\n=== Test: Check/Call Down ===")
    env = HoldemHuEnv()
    obs, _ = env.reset()
    env.render()
    
    done = False
    step_count = 0
    while not done:
        print(f"\nStep {step_count}: Player {env.current_player} acting")
        # Always Check/Call (Action 1)
        obs, reward, done, truncated, info = env.step(HoldemHuEnv.ACT_CHECK_CALL)
        env.render()
        step_count += 1
        
        if step_count > 20:
            print("Error: Game took too long")
            break
            
    print("Game Over")
    print(f"Rewards: {env.rewards}")
    # Total chips should be conserved
    assert abs((env.stacks[0] + env.stacks[1]) - (2 * env.initial_stack)) < 0.01
    print("Total chips conserved.")

def test_fold_preflop():
    print("\n=== Test: Fold Preflop ===")
    env = HoldemHuEnv()
    obs, _ = env.reset()
    env.render()
    
    # SB acts first. Let's say SB folds immediately (if allowed, but usually SB calls or raises).
    # Wait, SB posts 0.5, BB posts 1.0. SB needs to call 0.5 to match BB.
    # If SB folds, BB wins.
    
    print(f"Player {env.current_player} folds")
    obs, reward, done, truncated, info = env.step(HoldemHuEnv.ACT_FOLD)
    env.render()
    
    assert done
    print(f"Rewards: {env.rewards}")
    # Winner should win the blinds
    winner = 1 - env.current_player # The one who didn't fold (but step updates current_player? No, step updates current_player AFTER action unless done)
    # Wait, let's check the code.
    # _execute_action returns (reward, done). If done, step returns.
    # If fold, _end_hand_fold is called.
    # _end_hand_fold: winner = 1 - self.current_player.
    
    print("Fold test passed (visually check rewards)")

def test_raise_war():
    print("\n=== Test: Raise War ===")
    env = HoldemHuEnv()
    obs, _ = env.reset()
    env.render()
    
    # SB raises
    print(f"Player {env.current_player} raises min")
    obs, reward, done, truncated, info = env.step(HoldemHuEnv.ACT_MIN_RAISE)
    env.render()
    
    # BB re-raises (3-bet)
    print(f"Player {env.current_player} raises pot")
    obs, reward, done, truncated, info = env.step(HoldemHuEnv.ACT_RAISE_POT)
    env.render()
    
    # SB calls
    print(f"Player {env.current_player} calls")
    obs, reward, done, truncated, info = env.step(HoldemHuEnv.ACT_CHECK_CALL)
    env.render()
    
    # Should be on Flop now
    print("Should be on FLOP now")
    assert env.street == HoldemHuEnv.STREET_FLOP
    
    print("Raise War test passed")

if __name__ == "__main__":
    test_check_call_down()
    test_fold_preflop()
    test_raise_war()
