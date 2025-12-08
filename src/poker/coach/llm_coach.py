"""
LLM-powered poker coach that uses Ollama (local Llama models) for natural language recommendations.
Falls back to template-based recommendations if Ollama is unavailable.
"""
import os
from src.poker.coach.agentic_coach import AgenticCoach

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    print("Warning: Ollama package not installed. LLM coach will use template-based recommendations.")
    print("Install with: pip install ollama")
    print("Also make sure Ollama is running: ollama serve")

class LLMCoach:
    """
    LLM-powered coach that provides natural language recommendations
    with different risk tolerance levels.
    """
    
    RISK_LEVELS = {
        'conservative': {
            'name': 'Conservative',
            'description': 'Focus on value, avoid marginal spots',
            'prompt_modifier': 'You are a conservative poker player. Prioritize value and avoid marginal situations. Only recommend aggressive actions with strong hands.'
        },
        'moderate': {
            'name': 'Moderate',
            'description': 'Balanced approach, mix of value and bluffs',
            'prompt_modifier': 'You are a balanced poker player. Mix value betting with selective bluffs. Consider pot odds and position carefully.'
        },
        'aggressive': {
            'name': 'Aggressive',
            'description': 'Apply pressure, exploit weaknesses',
            'prompt_modifier': 'You are an aggressive poker player. Apply pressure and exploit opponent weaknesses. Be willing to bluff and value bet thinner.'
        }
    }
    
    def __init__(self, bot_policy=None, risk_tolerance='moderate', model='llama3.2', host='http://localhost:11434'):
        """
        Initialize LLM coach.
        
        Args:
            bot_policy: Bot's policy wrapper (for rollouts)
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
            model: Ollama model to use (default: llama3.2, also try: llama2, mistral, etc.)
            host: Ollama server host (default: http://localhost:11434)
        """
        self.base_coach = AgenticCoach(
            bot_policy=bot_policy,
            use_rollouts=bot_policy is not None,
            use_equity=True
        )
        
        self.risk_tolerance = risk_tolerance
        self.model = model
        self.host = host
        
        # Check if Ollama is available
        self.ollama_available = False
        if HAS_OLLAMA:
            try:
                # Test connection to Ollama
                ollama.list()  # This will raise an error if Ollama is not running
                self.ollama_available = True
                print(f"Ollama connected. Using model: {model}")
                # Check if model is available, if not suggest pulling it
                try:
                    ollama.show(model)
                except:
                    print(f"Warning: Model '{model}' not found. Run: ollama pull {model}")
                    print("Falling back to template-based recommendations.")
                    self.ollama_available = False
            except Exception as e:
                print(f"Warning: Could not connect to Ollama server at {host}")
                print(f"Error: {e}")
                print("Make sure Ollama is running: ollama serve")
                print("Or install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
                self.ollama_available = False
    
    def set_risk_tolerance(self, level):
        """Change risk tolerance level."""
        if level in self.RISK_LEVELS:
            self.risk_tolerance = level
        else:
            raise ValueError(f"Risk tolerance must be one of: {list(self.RISK_LEVELS.keys())}")
    
    def get_recommendation(self, env, n_rollouts=200, n_equity_samples=3000):
        """
        Get recommendation with LLM-powered explanation.
        
        Returns:
            dict with:
                - recommended_action: action_id
                - action_name: Human-readable name
                - explanation: LLM-generated or template explanation
                - tool_results: Dict of tool outputs (equity, pot_odds, etc.)
                - stats: Dict of key statistics
                - risk_analysis: Analysis for different risk levels
        """
        # Get base recommendation from AgenticCoach
        base_rec = self.base_coach.get_recommendation(env, n_rollouts, n_equity_samples)
        
        # Extract stats
        stats = self._extract_stats(env, base_rec)
        
        # Generate LLM recommendation if available
        if self.ollama_available:
            try:
                llm_explanation = self._get_llm_recommendation(env, base_rec, stats)
                base_rec['explanation'] = llm_explanation
            except Exception as e:
                print(f"Ollama API error: {e}. Using template-based recommendation.")
        
        # Add risk analysis for all levels
        base_rec['risk_analysis'] = self._get_risk_analysis(env, base_rec, stats)
        base_rec['stats'] = stats
        
        return base_rec
    
    def _extract_stats(self, env, base_rec):
        """Extract key statistics for display."""
        player = env.current_player
        tool_results = base_rec.get('tool_results', {})
        
        stats = {
            'pot': env.pot,
            'to_call': env.get_to_call() if hasattr(env, 'get_to_call') else abs(env.street_investment[0] - env.street_investment[1]),
            'my_stack': env.stacks[player],
            'opp_stack': env.stacks[1 - player],
            'street': ['Preflop', 'Flop', 'Turn', 'River'][env.street],
            'position': 'Button' if player == env.button else 'Big Blind',
            'spr': env.stacks[player] / (env.pot + 0.01),
        }
        
        # Add equity if available
        if 'equity' in tool_results:
            stats['equity'] = tool_results['equity'] * 100  # Convert to percentage
        else:
            stats['equity'] = None
        
        # Add pot odds
        if 'pot_odds' in tool_results:
            stats['pot_odds'] = tool_results['pot_odds'] * 100
            stats['required_equity'] = tool_results.get('required_equity', 0) * 100
        else:
            stats['pot_odds'] = 0
            stats['required_equity'] = 0
        
        # Add rollout EVs if available
        if 'rollout_evs' in tool_results:
            stats['rollout_evs'] = tool_results['rollout_evs']
        else:
            stats['rollout_evs'] = {}
        
        return stats
    
    def _get_llm_recommendation(self, env, base_rec, stats):
        """Get LLM-generated recommendation using Ollama."""
        player = env.current_player
        my_hand = env.hands[player]
        board = env.board
        
        # Build prompt
        hand_str = ' '.join([str(c) for c in my_hand])
        board_str = ' '.join([str(c) for c in board]) if board else "None"
        
        risk_prompt = self.RISK_LEVELS[self.risk_tolerance]['prompt_modifier']
        
        prompt = f"""You are a professional poker coach. {risk_prompt}

Current Situation:
- Street: {stats['street']}
- Your Hand: {hand_str}
- Board: {board_str}
- Pot: {stats['pot']:.1f} BB
- To Call: {stats['to_call']:.1f} BB
- Your Stack: {stats['my_stack']:.1f} BB
- Opponent Stack: {stats['opp_stack']:.1f} BB
- Position: {stats['position']}
- SPR (Stack-to-Pot Ratio): {stats['spr']:.2f}
"""
        
        if stats['equity'] is not None:
            prompt += f"- Your Equity: {stats['equity']:.1f}%\n"
            prompt += f"- Pot Odds: {stats['pot_odds']:.1f}%\n"
            prompt += f"- Required Equity: {stats['required_equity']:.1f}%\n"
            if stats['equity'] > stats['required_equity']:
                prompt += f"- Equity Edge: +{stats['equity'] - stats['required_equity']:.1f}%\n"
            else:
                prompt += f"- Equity Deficit: {stats['equity'] - stats['required_equity']:.1f}%\n"
        
        if stats['rollout_evs']:
            prompt += "\nRollout Expected Values:\n"
            for action_id, ev in sorted(stats['rollout_evs'].items(), key=lambda x: x[1], reverse=True):
                action_name = base_rec['action_name'] if action_id == base_rec['recommended_action'] else self.base_coach.ACTION_NAMES.get(action_id, f"Action {action_id}")
                prompt += f"  {action_name}: {ev:+.2f} BB\n"
        
        prompt += f"""
Legal Actions: {env._get_legal_actions()}

Provide a concise recommendation (2-3 sentences) that:
1. States the recommended action ({base_rec['action_name']})
2. Explains the key reasoning (equity, pot odds, position, SPR, etc.)
3. Mentions one strategic consideration

Be specific and actionable. Format as plain text, no markdown."""

        # Use Ollama API
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'num_predict': 200  # Max tokens
            }
        )
        
        return response['response'].strip()
    
    def _get_risk_analysis(self, env, base_rec, stats):
        """Generate recommendations for all risk tolerance levels."""
        analysis = {}
        
        for risk_level in self.RISK_LEVELS.keys():
            # Temporarily set risk tolerance
            old_risk = self.risk_tolerance
            self.risk_tolerance = risk_level
            
            # Get recommendation for this risk level
            # For now, use heuristic based on equity and pot odds
            action = self._recommend_for_risk_level(env, stats, risk_level)
            analysis[risk_level] = {
                'action': action,
                'action_name': self.base_coach.ACTION_NAMES.get(action, 'Unknown'),
                'reasoning': self._get_risk_reasoning(env, stats, risk_level, action)
            }
            
            # Restore risk tolerance
            self.risk_tolerance = old_risk
        
        return analysis
    
    def _recommend_for_risk_level(self, env, stats, risk_level):
        """Recommend action based on risk level."""
        legal_actions = env._get_legal_actions()
        equity = stats.get('equity', 0) / 100 if stats.get('equity') else None
        pot_odds = stats.get('pot_odds', 0) / 100
        
        if risk_level == 'conservative':
            # Conservative: Only value bet, fold marginal
            if equity and equity > pot_odds + 0.20:  # Strong equity edge
                if 5 in legal_actions:  # Half pot
                    return 5
                return 1  # Call
            elif equity and equity > pot_odds:
                return 1  # Call
            else:
                return 0  # Fold
        
        elif risk_level == 'moderate':
            # Moderate: Balanced
            if equity and equity > pot_odds + 0.10:
                if 5 in legal_actions:
                    return 5
                return 1
            elif equity and equity > pot_odds:
                return 1
            else:
                return 0
        
        else:  # aggressive
            # Aggressive: Apply pressure
            if equity and equity > pot_odds - 0.05:  # Even slight edge
                if 7 in legal_actions:  # Pot-sized
                    return 7
                elif 5 in legal_actions:
                    return 5
                return 1
            elif stats['spr'] < 5:  # Low SPR, commit
                return 1
            else:
                return 0
    
    def _get_risk_reasoning(self, env, stats, risk_level, action):
        """Get reasoning for risk level recommendation."""
        equity = stats.get('equity')
        pot_odds = stats.get('pot_odds', 0)
        
        if risk_level == 'conservative':
            if equity and equity > pot_odds:
                return f"Equity ({equity:.1f}%) exceeds pot odds ({pot_odds:.1f}%) - profitable call."
            return "Conservative fold - not enough equity edge."
        
        elif risk_level == 'moderate':
            if equity and equity > pot_odds:
                return f"Balanced play - equity ({equity:.1f}%) justifies action."
            return "Moderate fold - marginal spot."
        
        else:  # aggressive
            if equity and equity > pot_odds - 5:
                return f"Aggressive play - apply pressure with {equity:.1f}% equity."
            return "Aggressive call - exploit opponent weakness."

