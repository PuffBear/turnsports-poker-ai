
import numpy as np
import json
import os

class LLMAgent:
    """
    Poker agent controlled by an LLM based on user instructions.
    """
    
    def __init__(self, model_name="gpt-4", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Action map
        self.action_names = [
            "FOLD",
            "CHECK_CALL",
            "MIN_RAISE",
            "RAISE_QUARTER_POT",
            "RAISE_THIRD_POT",
            "RAISE_HALF_POT",
            "RAISE_THREE_QUARTER_POT",
            "RAISE_POT",
            "ALL_IN"
        ]

    def get_action(self, game_state, user_instruction, legal_actions):
        """
        Decide an action based on game state and user instruction.
        
        Args:
            game_state (dict): Dictionary containing game info (cards, pot, history, etc.)
            user_instruction (str): Strategy instruction from the user (e.g. "Play tight")
            legal_actions (list): List of legal action IDs
            
        Returns:
            action_id (int): The chosen action
            explanation (str): Reasoning for the action
        """
        
        # Construct the prompt
        prompt = self._construct_prompt(game_state, user_instruction, legal_actions)
        
        # Call LLM (Mock for now if no API key)
        if not self.api_key:
            return self._mock_response(legal_actions)
            
        try:
            # This is where the actual API call would go
            # response = openai.ChatCompletion.create(...)
            # For now, we'll just print what we WOULD send and return a mock
            print(f"\n[LLM Agent] Constructing prompt for {self.model_name}...")
            # print(prompt) 
            return self._mock_response(legal_actions)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._mock_response(legal_actions)

    def _construct_prompt(self, game_state, user_instruction, legal_actions):
        """Build the prompt for the LLM."""
        
        legal_action_names = [f"{i}: {self.action_names[i]}" for i in legal_actions]
        
        prompt = f"""
You are a professional poker player. You are playing Heads-Up No-Limit Texas Hold'em.
Your goal is to play optimally while following the user's strategic instructions.

Current Game State:
- Your Hand: {game_state['hand']}
- Board: {game_state['board']}
- Pot: {game_state['pot']} BB
- Your Stack: {game_state['my_stack']} BB
- Opponent Stack: {game_state['opp_stack']} BB
- Street: {game_state['street']}
- Position: {'Button (IP)' if game_state['is_button'] else 'Big Blind (OOP)'}
- To Call: {game_state['to_call']} BB

User Instruction: "{user_instruction}"

Legal Actions:
{', '.join(legal_action_names)}

Task:
1. Analyze the situation based on the game state and user instruction.
2. Choose the best action from the legal actions.
3. Provide a brief explanation of your reasoning.

Output Format (JSON):
{{
    "action_id": <int>,
    "reasoning": "<string>"
}}
"""
        return prompt

    def _mock_response(self, legal_actions):
        """Return a valid random action for testing."""
        import random
        action = random.choice(legal_actions)
        return action, f"I'm picking {self.action_names[action]} because I don't have a real brain yet (API key missing). But if I did, I'd follow your advice!"
