# Hold'em Design Notes

## Architecture Overview

This document describes the design of the Heads-Up No-Limit Texas Hold'em (HU NLHE) poker bot and agentic coach system.

## 1. Environment Design

### Game Specifications
- **Players**: 2 (Heads-Up)
- **Starting Stacks**: 100BB each
- **Blinds**: 0.5BB (SB) / 1BB (BB)
- **Streets**: Preflop, Flop, Turn, River
- **Betting**: No-Limit with bet abstraction

### Bet Abstraction

To manage the enormous action space of NLHE, we use a discrete bet abstraction:

| Action ID | Action | Description |
|-----------|--------|-------------|
| 0 | Fold | Give up the hand |
| 1 | Check/Call | Check (no bet) or call current bet |
| 2 | Min Raise | Minimum legal raise |
| 3 | 1/4 Pot | Raise 25% of pot |
| 4 | 1/3 Pot | Raise 33% of pot |
| 5 | 1/2 Pot | Raise 50% of pot |
| 6 | 3/4 Pot | Raise 75% of pot |
| 7 | Pot | Raise 100% of pot |
| 8 | All-In | Bet entire stack |

**Legality Filtering**: Not all actions are legal at all times. The environment filters actions based on:
- Whether there's a bet to call (fold only legal if facing bet)
- Stack sizes (can't raise more than stack)
- Minimum raise rules (must raise at least min-raise amount)

### State Representation

The observation vector (200 dimensions) encodes:

1. **Hole Cards** (34 dims)
   - 2 cards × 17 features (13 rank one-hot + 4 suit one-hot)

2. **Board Cards** (85 dims)
   - 5 cards × 17 features (padded with zeros if not dealt yet)

3. **Pot Information** (10 dims)
   - Current pot size (normalized)
   - Amount to call (normalized)
   - Hero stack (normalized)
   - Villain stack (normalized)
   - SPR (Stack-to-Pot Ratio)
   - Last bet size / pot ratio

4. **Street** (4 dims)
   - One-hot: [Preflop, Flop, Turn, River]

5. **Position** (2 dims)
   - Button/SB indicator
   - In-position indicator

6. **Betting History** (65 dims)
   - Abstracted pattern of previous actions
   - Preflop: limped, raised, 3-bet, etc.
   - Postflop: checked, bet-called, raised, etc.

### Reward Structure

- **Sparse rewards**: Only non-zero at terminal states
- **Reward = Net chips won/lost** (in BB relative to starting stack)
- Example: Win 10BB pot → reward = +10
- Example: Lose 5BB → reward = -5

## 2. DQN Agent Architecture

### Network Structure

```
Input (200) → FC(256) → ReLU → FC(256) → ReLU → FC(256) → ReLU → Output(9)
```

- **Input**: State vector (200 dims)
- **Hidden Layers**: 3 layers of 256 neurons with ReLU activation
- **Output**: Q-values for 9 actions

### Training Algorithm

**Double DQN** with experience replay:

1. **Experience Replay**:
   - Buffer size: 100k transitions
   - Batch size: 64
   - Stores: (state, action, reward, next_state, done)

2. **Target Network**:
   - Separate network for computing target Q-values
   - Updated every 100 episodes
   - Stabilizes training

3. **Epsilon-Greedy Exploration**:
   - Start: ε = 1.0 (100% random)
   - End: ε = 0.1 (10% random)
   - Decay: 0.9995 per step

4. **Loss Function**:
   - MSE between predicted Q(s,a) and target Q
   - Target: r + γ * max Q(s', a')

### Training Phases

**Phase A: vs Rule-Based Opponents** (Current)
- Mix of Random, TAG (Tight-Aggressive), LAG (Loose-Aggressive)
- Teaches basic concepts: value betting, hand strength, position

**Phase B: Self-Play** (Future)
- Maintain pool of past agent snapshots
- Sample opponent from pool each episode
- Converges toward Nash equilibrium

**Phase C: Exploitation Training** (Future)
- Train against specific opponent types
- Learn exploitative strategies

## 3. Opponent Models

### Random Opponent
- Uniformly samples from legal actions
- Baseline for evaluation

### TAG (Tight-Aggressive) Opponent
- **Preflop**: Plays top 20% of hands, raises with strong hands
- **Postflop**: C-bets when hitting flop, folds weak hands
- **Sizing**: Prefers 1/2 pot and pot-sized bets

### LAG (Loose-Aggressive) Opponent
- **Preflop**: Plays wide range (50%+)
- **Postflop**: Bets frequently, over-bluffs
- **Sizing**: Variable, sometimes overlarge

## 4. Agentic Coach Design

The coach is an **agentic AI** that uses multiple tools to analyze poker situations.

### Tool 1: Equity Estimator

**Purpose**: Estimate win probability (equity) against villain's range

**Algorithm**: Monte Carlo simulation
1. Deal random villain hands (from remaining deck)
2. Complete the board with random cards
3. Evaluate both hands
4. Equity = (wins + 0.5 * ties) / simulations

**Parameters**:
- `n_samples`: 3000-5000 for accuracy vs. speed tradeoff
- `villain_range`: Currently "random", can extend to ranges (top 20%, etc.)

**Example Output**: "Your equity: ~68%"

### Tool 2: Pot Odds Calculator

**Purpose**: Determine if calling is profitable based on pot odds

**Formula**: 
```
Pot Odds = Amount to Call / (Pot + Amount to Call)
Required Equity = Pot Odds
```

**Example**: 
- Pot = 12BB, To Call = 6BB
- Pot Odds = 6 / (12 + 6) = 0.33 (33%)
- Need 33% equity to break even

### Tool 3: Rollout Simulator

**Purpose**: Estimate EV of different actions by simulation

**Algorithm**:
1. Clone current game state
2. Force hero to take action X
3. Let bot play rest of hand
4. Record outcome (chips won/lost)
5. Repeat N times, compute average EV

**Parameters**:
- `n_episodes`: 200-300 rollouts per action
- `candidate_actions`: 3-4 most interesting actions (fold, call, raise, all-in)

**Example Output**:
```
Fold: 0 BB
Call: +0.3 BB
1/2 Pot Raise: +0.5 BB
All-In: -0.2 BB
```

### Coach Decision Logic

```python
def get_recommendation(env):
    # 1. Summarize state
    summary = summarize_state(env)
    
    # 2. Calculate equity (if postflop)
    equity = estimate_equity(my_hand, board)
    
    # 3. Calculate pot odds
    pot_odds = compute_pot_odds(pot, to_call)
    
    # 4. Rollout candidate actions
    evs = compare_actions(env, [fold, call, raise_half, all_in])
    
    # 5. Recommend highest EV action
    recommended = max(evs, key=evs.get)
    
    # 6. Generate explanation
    explanation = generate_explanation(...)
    
    return recommendation
```

### Explanation Generation

The coach provides multi-level explanations:

1. **Recommended Action**: "1/2 Pot Raise"

2. **Equity Analysis**:
   - "Your equity: ~68%"
   - "Pot odds: 25% (need 25% equity to call)"
   - "You have +43% equity edge"

3. **Rollout EVs**:
   - Fold: 0 BB
   - Call: +0.3 BB
   - 1/2 Pot: +0.5 BB (recommended)

4. **Strategic Reasoning**:
   - "Low SPR - consider commitment"
   - "You have position - can control pot size"
   - "Exploitative vs this bot profile"

## 5. GUI Design

### Layout

```
┌─────────────────────────────┬─────────────────────────┐
│     GAME STATE              │    AI COACH             │
│                             │                         │
│ Board: K♦ 7♠ 2♦             │ Recommended: 1/2 Pot    │
│ Pot: 12 BB                  │                         │
│                             │ Analysis:               │
│ Your Hand: K♣ Q♦            │ • Equity: ~68%          │
│ Your Stack: 88 BB           │ • Pot odds: 25%         │
│                             │ • +43% equity edge      │
│ Bot Stack: 92 BB            │                         │
│                             │ Rollout EVs:            │
│ Action History:             │   1/2 Pot: +0.5 BB      │
│ ┌─────────────────────────┐ │   Call: +0.3 BB         │
│ │ New hand started        │ │   Fold: 0 BB            │
│ │ Bot: 1/2 Pot Raise      │ │                         │
│ │                         │ │ • Low SPR situation     │
│ └─────────────────────────┘ │ • You have position     │
│                             │                         │
│ [Fold] [Call] [1/4] [1/3]   │ [Get Coach Advice]      │
│ [1/2]  [3/4]  [Pot] [All-In]│                         │
│                             │                         │
│        [New Hand]           │                         │
└─────────────────────────────┴─────────────────────────┘
```

### Features

1. **Real-time Game State**: Cards, pot, stacks update automatically
2. **Action Buttons**: Only legal actions enabled
3. **History Log**: Scrollable action history
4. **Coach Panel**: 
   - On-demand advice (click button)
   - Shows reasoning and tool outputs
   - Updates each player turn
5. **Bot Delays**: 500ms artificial delay for readability

## 6. State Machine

```
[Reset] → Blinds Posted → Preflop Betting
              ↓
         Deal Flop → Flop Betting
              ↓
         Deal Turn → Turn Betting
              ↓
         Deal River → River Betting
              ↓
          Showdown → [Terminal]
              ↑
          Fold/All-in → [Terminal]
```

### Betting Round Logic

Within each street:
1. First player to act takes action
2. Action passes to other player
3. Round ends when:
   - Both players checked
   - One player called a bet/raise
   - One player folded
   - One player all-in

**Special Cases**:
- Preflop: SB acts first, BB has "option" (can raise after SB calls)
- Postflop: BB acts first (Button is in position)
- All-in: Runout remaining cards, go to showdown

## 7. Future Extensions

### LLM Integration

Instead of hardcoded explanations, use an LLM:

```python
def get_llm_explanation(state, equity, pot_odds, evs):
    prompt = f"""
    You are a poker coach. Analyze this situation:
    
    State: {state}
    Hero Equity: {equity}
    Pot Odds: {pot_odds}
    Rollout EVs: {evs}
    
    Provide a concise recommendation and explain your reasoning.
    """
    
    response = llm.generate(prompt)
    return response
```

### Range-Based Equity

Currently equity is vs. random hands. Extend to ranges:
- "Top 20% of hands"
- "All pairs and high cards"
- Custom range strings (e.g., "22+, AK, AQ")

### Bet Sizing Optimization

Instead of fixed abstractions, learn continuous bet sizes:
- Use actor-critic (A3C/PPO) instead of DQN
- Actor outputs bet size in [0, stack]
- More realistic poker

### CFR+ for Game Theory

DQN learns exploitative play. For game-theoretic optimality:
- Implement CFR (Counterfactual Regret Minimization)
- Converges to Nash equilibrium
- Unexploitable strategy

---

**Design Philosophy**: Balance complexity and learnability. Start with abstractions (discrete bets), layer on sophistication (continuous sizes, ranges, CFR) as needed.
