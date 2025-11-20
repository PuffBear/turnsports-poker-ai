# Agentic AI Coach Architecture

## Overview

The **Agentic AI Coach** is a tool-using AI system that helps players make better poker decisions. Unlike a simple rule-based advisor, the coach actively **calls tools** to gather information and synthesizes that information into actionable advice.

## What Makes It "Agentic"?

An agentic AI system has these properties:

1. **Goal-Oriented**: Has a clear objective (help player win)
2. **Tool-Using**: Calls external tools/functions to gather info
3. **Reasoning**: Combines tool outputs to make decisions
4. **Explainable**: Provides human-readable explanations

### Comparison

| Traditional Bot | Rule-Based Advisor | Agentic Coach |
|----------------|-------------------|---------------|
| Plays poker | Gives fixed tips | Analyzes situations |
| No explanation | Simple rules | Multi-tool reasoning |
| "I fold" | "Always fold weak hands" | "Fold - 32% equity vs 40% required, -0.5BB EV" |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  AGENTIC COACH                      │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         Decision Engine                      │  │
│  │  1. Observe state                            │  │
│  │  2. Call tools                               │  │
│  │  3. Synthesize results                       │  │
│  │  4. Generate recommendation + explanation    │  │
│  └──────────────────────────────────────────────┘  │
│                       │                             │
│         ┌─────────────┼─────────────┐              │
│         ▼             ▼             ▼               │
│   ┌─────────┐  ┌──────────┐  ┌──────────┐         │
│   │ Equity  │  │   Pot    │  │ Rollout  │         │
│   │  Tool   │  │   Odds   │  │   Tool   │         │
│   └─────────┘  └──────────┘  └──────────┘         │
│        │             │              │               │
│        └─────────────┴──────────────┘              │
│                       │                             │
│                       ▼                             │
│              ┌─────────────────┐                   │
│              │  Explanation    │                   │
│              │   Generator     │                   │
│              └─────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

## Tools

### Tool 1: State Summarizer

**Input**: Raw environment state  
**Output**: Human-readable summary + structured data

```python
summary = summarize_state(env)
# Returns:
{
    'street': 'Flop',
    'hand': 'KcQd',
    'board': 'Kd7s2d',
    'pot': 12.0,
    'my_stack': 88.0,
    'opp_stack': 92.0,
    'to_call': 6.0,
    'position': 'Button/SB',
    'in_position': True,
    'spr': 7.33,
    'pot_odds': 0.33,
    'text': '...'
}
```

**Why It's Useful**: Converts low-level game state into high-level concepts the coach (and player) can reason about.

### Tool 2: Equity Estimator

**Input**: Hero hand, board, villain range  
**Output**: Win probability (equity)

```python
equity = estimate_equity(
    my_hand=[Kc, Qd],
    board=[Kd, 7s, 2d],
    villain_range='random',
    n_samples=5000
)
# Returns: 0.68 (68% equity)
```

**Algorithm**: Monte Carlo simulation
1. Sample villain hand from range
2. Deal remaining board cards
3. Evaluate hands
4. Repeat N times
5. Equity = (wins + 0.5*ties) / N

**Why It's Useful**: Fundamental question in poker - "How often do I win?"

### Tool 3: Pot Odds Calculator

**Input**: Pot size, amount to call  
**Output**: Break-even equity required

```python
pot_odds = compute_pot_odds(pot=12, to_call=6)
# Returns: 0.33 (need 33% equity to call profitably)
```

**Formula**: 
```
Pot Odds = Call / (Pot + Call)
```

**Example**:
- Pot = 12BB, Call = 6BB
- Odds = 6 / (12 + 6) = 0.33
- Need 33% equity to break even

**Why It's Useful**: Immediate profitability check - if equity > pot odds, calling is +EV.

### Tool 4: Rollout Simulator

**Input**: Environment, action, bot policy  
**Output**: Expected value (EV) of taking that action

```python
ev = rollout_action(
    env=current_env,
    action_id=5,  # 1/2 pot raise
    bot_policy=trained_bot,
    n_episodes=200
)
# Returns: +0.5 (average +0.5BB from this action)
```

**Algorithm**:
1. Clone environment
2. Force hero to take action
3. Let bot play optimally for rest of hand
4. Record outcome (± chips)
5. Repeat N times, average result

**Why It's Useful**: Estimates EV of complex lines. More accurate than static equity, considers opponent tendencies.

**Comparison**:
```
Fold:        0.00 BB (guaranteed)
Call:       +0.30 BB (on average)
1/2 Pot:    +0.50 BB (best action)
All-in:     -0.20 BB (overplay)
```

## Decision Algorithm

The coach follows this workflow:

```python
def get_recommendation(env):
    # Step 1: Observe
    summary = summarize_state(env)
    legal_actions = env.get_legal_actions()
    
    # Step 2: Call Tools
    equity = estimate_equity(my_hand, board) if board else None
    pot_odds = compute_pot_odds(summary['pot'], summary['to_call'])
    
    candidate_actions = select_candidates(legal_actions)  # e.g., [fold, call, 1/2pot]
    rollout_evs = compare_actions(env, candidate_actions, bot_policy)
    
    # Step 3: Synthesize
    # Choose action with highest EV
    recommended_action = max(rollout_evs, key=rollout_evs.get)
    
    # Step 4: Explain
    explanation = generate_explanation(
        action=recommended_action,
        equity=equity,
        pot_odds=pot_odds,
        evs=rollout_evs,
        summary=summary
    )
    
    return {
        'recommended_action': recommended_action,
        'action_name': ACTION_NAMES[recommended_action],
        'explanation': explanation,
        'tool_results': {
            'equity': equity,
            'pot_odds': pot_odds,
            'rollout_evs': rollout_evs
        }
    }
```

## Explanation Generation

The coach generates **layered explanations**:

### Level 1: Recommendation
```
Recommended: 1/2 Pot Raise
```

### Level 2: Tool Results
```
• Your equity: ~68%
• Pot odds: 33% (need 33% equity to call)
• You have profitable calling odds (+35% equity edge)

Rollout EVs:
  1/2 Pot Raise: +0.5 BB
  Call: +0.3 BB
  Fold: 0 BB
```

### Level 3: Strategic Reasoning
```
• Low SPR situation - consider commitment
• You have position - can control pot size
```

### Full Example

```
Recommended: 1/2 Pot Raise

• Your equity: ~68%
• Pot odds: 33% (need 33% equity to call)
• You have profitable calling odds (+35% equity edge)

Rollout EVs:
  1/2 Pot Raise: +0.5 BB
  Call: +0.3 BB
  Fold: 0 BB

• Low SPR situation - consider commitment
• You have position - can control pot size
```

## LLM Integration (Future)

Currently, explanations are template-based. Future version uses LLM:

```python
def get_llm_recommendation(state, tool_results):
    prompt = f"""
    You are a professional poker coach analyzing a hand.
    
    Situation:
    {state['text']}
    
    Tool Analysis:
    - Hero Equity: {tool_results['equity']*100:.1f}%
    - Pot Odds: {tool_results['pot_odds']*100:.1f}%
    - Rollout EVs: {tool_results['rollout_evs']}
    
    Provide:
    1. Recommended action
    2. Brief explanation (2-3 sentences)
    3. Key strategic consideration
    
    Be concise and actionable.
    """
    
    response = llm.complete(prompt)
    return response
```

**Benefits**:
- More natural language
- Context-aware explanations
- Can ask follow-up questions

**Example LLM Output**:
```
Recommended: 1/2 Pot Raise

You flopped top pair with a flush draw, giving you 68% equity - well 
above the 33% needed to call. The rollout simulator shows raising 
1/2 pot has the highest EV (+0.5BB) because it charges worse hands 
while protecting against the diamond draw. 

With only 7x SPR and position, you want to build the pot now while 
you're ahead. This sizing accomplishes that without over-committing.
```

## Agentic Workflow

The key difference from traditional systems:

**Traditional Recommender**:
```
Input → Fixed Rules → Output
```

**Agentic Coach**:
```
Input → Plan → Call Tools → Synthesize → Explain → Output
         ↑                      ↓
         └──── Adapt Based on Results
```

### Example: Adapting to Situation

**Scenario 1: Preflop (no board)**
- Skip equity estimation (no board yet)
- Use preflop hand strength heuristics
- Focus on position, stack sizes

**Scenario 2: River (all cards dealt)**
- Full equity calculation
- Detailed rollout (fewer unknown cards)
- Consider pot commitment

**Scenario 3: Facing All-In**
- Only 2 choices: call or fold
- Skip raise rollouts
- Focus heavily on equity vs pot odds

## Tool Selection Strategy

The coach doesn't always call all tools (efficiency):

```python
def select_tools(state):
    tools = ['pot_odds']  # Always calculate
    
    if len(state['board']) > 0:
        tools.append('equity')  # Only postflop
    
    if state['spr'] < 10 and bot_policy:
        tools.append('rollout')  # Deep enough for meaningful lines
    
    return tools
```

## Evaluation Metrics

How do we measure coach quality?

### Accuracy
**Metric**: % of times coach recommends GTO-optimal action  
**Baseline**: Compare to solver (e.g., PioSolver)

### Explanation Quality (Human Eval)
- Clarity (1-5 stars)
- Actionability (1-5 stars)
- Correctness (1-5 stars)

### User Performance
**Metric**: Player win rate with vs. without coach  
**Experiment**: A/B test players, measure bb/100

### Speed
**Metric**: Time to recommendation  
**Target**: < 3 seconds for real-time play

## Challenges & Solutions

### Challenge 1: Rollout Speed
**Problem**: 200 rollouts × 3 actions = 600 simulations → slow  
**Solution**: 
- Parallel rollouts
- Reduce n_episodes for fast feedback (100 instead of 200)
- Cache common positions

### Challenge 2: Equity Accuracy
**Problem**: 5000 samples = 95% CI of ±1.4%  
**Solution**:
- Adaptive sampling (more samples when close)
- Use lookup tables for common flops
- GPU acceleration

### Challenge 3: Range Estimation
**Problem**: Don't know villain's exact range  
**Solution**:
- Start with "random" range
- Learn opponent model from history
- Ask user to input range (advanced mode)

### Challenge 4: Explanation Overload
**Problem**: Too much info overwhelms player  
**Solution**:
- Progressive disclosure (expand for details)
- Confidence-weighted display (only show close EVs)
- Beginner vs Expert mode

## Extensibility

The coach is designed to be modular. Adding new tools:

```python
class NewTool:
    def analyze(self, state):
        # Custom analysis
        return result

# Register tool
coach.register_tool('new_tool', NewTool())

# Coach will automatically use it
recommendation = coach.get_recommendation(env)
```

**Example Extensions**:
- **Blockers Tool**: Analyze card removal effects
- **Range Tool**: Construct villain ranges from actions
- **History Tool**: Analyze past hands for patterns
- **GTO Tool**: Compare to solver baseline

## Conclusion

The Agentic Coach demonstrates key principles of modern AI systems:

1. **Modularity**: Tools are independent, composable
2. **Transparency**: All reasoning is explainable
3. **Adaptability**: Behavior changes based on context
4. **Human-Centered**: Designed to teach, not just advise

This architecture can extend beyond poker to any strategic decision-making domain (chess, trading, business strategy, etc.).

---

**Next Steps**:
- [ ] Add LLM for natural language
- [ ] Implement range-based equity
- [ ] Build opponent modeling tool
- [ ] Create interactive explanation UI
- [ ] Multi-agent coach (different coaching styles)
