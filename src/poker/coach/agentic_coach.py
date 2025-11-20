from src.poker.coach.state_summary import summarize_state
from src.poker.coach.equity_tools import estimate_equity, compute_pot_odds
from src.poker.coach.rollout_tools import compare_actions

class AgenticCoach:
    """
    AI Coach that uses tools (equity calculation, rollouts) to recommend actions.
    Can be extended to use LLM for natural language explanations.
    """
    
    ACTION_NAMES = {
        0: "Fold",
        1: "Check/Call",
        2: "Min Raise",
        3: "1/4 Pot Raise",
        4: "1/3 Pot Raise",
        5: "1/2 Pot Raise",
        6: "3/4 Pot Raise",
        7: "Pot Raise",
        8: "All-In"
    }
    
    def __init__(self, bot_policy=None, use_rollouts=True, use_equity=True):
        """
        Args:
            bot_policy: Bot's policy wrapper (for rollouts)
            use_rollouts: Whether to use rollout simulations
            use_equity: Whether to calculate equity
        """
        self.bot_policy = bot_policy
        self.use_rollouts = use_rollouts
        self.use_equity = use_equity
    
    def get_recommendation(self, env, n_rollouts=200, n_equity_samples=3000):
        """
        Analyze current state and recommend an action.
        
        Returns:
            dict with:
                - recommended_action: action_id
                - action_name: Human-readable name
                - explanation: Text explanation
                - tool_results: Dict of tool outputs
        """
        player = env.current_player
        my_hand = env.hands[player]
        board = env.board
        legal_actions = env._get_legal_actions()
        
        # Get state summary
        summary = summarize_state(env)
        
        # Initialize results
        tool_results = {}
        
        # Calculate equity
        equity = None
        if self.use_equity and len(board) > 0:
            equity = estimate_equity(my_hand, board, n_samples=n_equity_samples)
            tool_results['equity'] = equity
        
        # Calculate pot odds
        pot_odds = compute_pot_odds(summary['pot'], summary['to_call'])
        tool_results['pot_odds'] = pot_odds
        required_equity = pot_odds
        tool_results['required_equity'] = required_equity
        
        # Rollout simulations
        rollout_evs = {}
        if self.use_rollouts and self.bot_policy:
            # Select candidate actions to rollout
            candidates = self._select_candidate_actions(legal_actions, env)
            rollout_evs = compare_actions(env, candidates, self.bot_policy, n_rollouts)
            tool_results['rollout_evs'] = rollout_evs
        
        # Make recommendation
        if rollout_evs:
            # Recommend action with highest EV
            recommended_action = max(rollout_evs, key=rollout_evs.get)
            best_ev = rollout_evs[recommended_action]
        else:
            # Fallback to simple heuristic
            recommended_action = self._heuristic_action(summary, equity, pot_odds, legal_actions)
            best_ev = None
        
        # Generate explanation
        explanation = self._generate_explanation(
            recommended_action, summary, equity, pot_odds, rollout_evs, best_ev
        )
        
        return {
            'recommended_action': recommended_action,
            'action_name': self.ACTION_NAMES[recommended_action],
            'explanation': explanation,
            'tool_results': tool_results,
            'summary': summary
        }
    
    def _select_candidate_actions(self, legal_actions, env):
        """Select 2-4 most interesting actions to rollout."""
        candidates = []
        
        # Always consider fold and call
        if 0 in legal_actions:
            candidates.append(0)
        if 1 in legal_actions:
            candidates.append(1)
        
        # Add 1-2 raise sizes
        raise_actions = [a for a in legal_actions if a >= 2]
        if raise_actions:
            # Add medium raise (half pot or pot)
            if 5 in raise_actions:  # Half pot
                candidates.append(5)
            elif 7 in raise_actions:  # Pot
                candidates.append(7)
            
            # Add all-in if available
            if 8 in raise_actions:
                candidates.append(8)
        
        return candidates[:4]  # Limit to 4 for speed
    
    def _heuristic_action(self, summary, equity, pot_odds, legal_actions):
        """Simple heuristic when rollouts not available."""
        # If we have strong equity relative to pot odds, call/raise
        if equity and pot_odds:
            if equity > pot_odds + 0.15:  # Good equity margin
                # Raise if possible
                if 5 in legal_actions:  # Half pot
                    return 5
                return 1  # Call
            elif equity > pot_odds:
                return 1  # Call
            else:
                return 0  # Fold
        
        # Default to call/check
        return 1 if 1 in legal_actions else 0
    
    def _generate_explanation(self, action, summary, equity, pot_odds, rollout_evs, best_ev):
        """Generate human-readable explanation."""
        lines = []
        
        lines.append(f"Recommended: {self.ACTION_NAMES[action]}")
        lines.append("")
        
        # Equity analysis
        if equity is not None:
            lines.append(f"• Your equity: ~{equity*100:.1f}%")
            if pot_odds > 0:
                lines.append(f"• Pot odds: {pot_odds*100:.1f}% (need {pot_odds*100:.1f}% equity to call)")
                if equity > pot_odds:
                    lines.append(f"• You have profitable calling odds (+{(equity-pot_odds)*100:.1f}% equity edge)")
                else:
                    lines.append(f"• Calling is -EV ({(equity-pot_odds)*100:.1f}% equity deficit)")
        
        # Rollout analysis
        if rollout_evs:
            lines.append("")
            lines.append("Rollout EVs:")
            for act_id in sorted(rollout_evs.keys(), key=lambda x: rollout_evs[x], reverse=True):
                ev = rollout_evs[act_id]
                name = self.ACTION_NAMES[act_id]
                lines.append(f"  {name}: {ev:+.2f}BB")
        
        # Strategic reasoning
        lines.append("")
        if summary['spr'] < 3:
            lines.append("• Low SPR situation - consider commitment")
        elif summary['spr'] > 10:
            lines.append("• Deep stacks - room for maneuvering")
        
        if summary['in_position']:
            lines.append("• You have position - can control pot size")
        else:
            lines.append("• Out of position - proceed with caution")
        
        return "\n".join(lines)
