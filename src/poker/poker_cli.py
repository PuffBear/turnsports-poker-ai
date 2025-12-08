
import sys
import os
import time
import random

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.agents.cfr_agent import CFRAgent
from src.poker.agents.llm_agent import LLMAgent

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_readable_state(env, player_idx):
    """Extract readable state for the LLM/User."""
    street_names = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
    
    # Get cards as strings
    my_hand = [str(c) for c in env.hands[player_idx]]
    board = [str(c) for c in env.board]
    
    # Calculate amounts
    pot = env.pot
    my_stack = env.stacks[player_idx]
    opp_stack = env.stacks[1 - player_idx]
    
    # Investment
    my_invest = env.street_investment[player_idx]
    opp_invest = env.street_investment[1 - player_idx]
    to_call = max(0, opp_invest - my_invest)
    
    return {
        'hand': my_hand,
        'board': board,
        'pot': round(pot, 2),
        'my_stack': round(my_stack, 2),
        'opp_stack': round(opp_stack, 2),
        'street': street_names[env.street],
        'is_button': (player_idx == env.button),
        'to_call': round(to_call, 2)
    }

def print_game_state(state, player_idx):
    print("\n" + "="*40)
    print(f"   POKER BATTLE: HERO (P{player_idx}) vs CFR BOT")
    print("="*40)
    
    print(f"\nSTREET: {state['street']}")
    print(f"POT:    {state['pot']} BB")
    print(f"BOARD:  {state['board']}")
    
    print("\nOPPONENT (CFR Bot)")
    print(f"Stack: {state['opp_stack']} BB")
    print(f"Cards: [??, ??]")
    
    print("\nHERO (You + LLM)")
    print(f"Stack: {state['my_stack']} BB")
    print(f"Cards: {state['hand']}")
    print(f"Pos:   {'BUTTON' if state['is_button'] else 'BB'}")
    
    if state['to_call'] > 0:
        print(f"\nTO CALL: {state['to_call']} BB")
    print("="*40 + "\n")

def main():
    # Initialize
    env = HoldemHuEnv()
    cfr_agent = CFRAgent() # Untrained for now, plays randomly/uniformly
    llm_agent = LLMAgent()
    
    print("Welcome to the LLM Poker CLI!")
    print("You give instructions, the LLM plays the hand.")
    input("Press Enter to start...")
    
    obs, _ = env.reset()
    
    # Determine who is who
    # In this env, P0 and P1 switch positions (button) each hand?
    # Actually env.button determines who is SB/Button.
    # Let's fix Hero as P0 for simplicity in tracking, but P0 might be SB or BB.
    hero_idx = 0
    opp_idx = 1
    
    running = True
    while running:
        clear_screen()
        
        # Check if hand is done
        if env.done:
            print("\n--- HAND OVER ---")
            print(f"Board: {[str(c) for c in env.board]}")
            print(f"Hero Hand: {[str(c) for c in env.hands[hero_idx]]}")
            print(f"Opp Hand:  {[str(c) for c in env.hands[opp_idx]]}")
            print(f"Result: {env.rewards}")
            
            if env.rewards[hero_idx] > 0:
                print("YOU WON!")
            elif env.rewards[hero_idx] < 0:
                print("YOU LOST.")
            else:
                print("CHOP POT.")
                
            play_again = input("\nPlay another hand? (y/n): ")
            if play_again.lower() != 'y':
                break
            
            env.reset()
            continue
            
        # Current player
        current_player = env.current_player
        
        if current_player == hero_idx:
            # HERO TURN
            state = get_readable_state(env, hero_idx)
            print_game_state(state, hero_idx)
            
            legal_actions = env._get_legal_actions()
            action_names = llm_agent.action_names
            print("Legal Actions:", [f"{i}:{action_names[i]}" for i in legal_actions])
            
            # Get User Instruction
            instruction = input("\n>> Enter instruction for LLM (e.g. 'play aggressive', 'fold trash'): ")
            if not instruction:
                instruction = "Play standard"
            
            # Get LLM Action
            print("\nThinking...")
            action, reasoning = llm_agent.get_action(state, instruction, legal_actions)
            
            print(f"\n[LLM] I choose: {action_names[action]}")
            print(f"[LLM] Reasoning: {reasoning}")
            
            input("\nPress Enter to execute...")
            env.step(action)
            
        else:
            # OPPONENT TURN
            # print("\nOpponent is thinking...")
            # time.sleep(0.5)
            
            legal_actions = env._get_legal_actions()
            info_set = cfr_agent._get_info_set(env, opp_idx)
            
            # Use average strategy (Nash approximation)
            # If untrained, this will be uniform random
            strategy = cfr_agent.get_average_strategy(info_set, legal_actions)
            action = cfr_agent.get_action(info_set, legal_actions) # This samples from strategy
            
            # print(f"Opponent checks/bets...")
            env.step(action)

if __name__ == "__main__":
    main()
