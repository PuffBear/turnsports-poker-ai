# CFR Poker Agent - System Architecture Deep Dive

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [CFR Algorithm Deep Dive](#cfr-algorithm-deep-dive)
5. [GUI Integration](#gui-integration)
6. [File Structure](#file-structure)

---

## System Overview

Your poker bot system consists of **5 major subsystems**:

```
┌─────────────────────────────────────────────────────────────┐
│                    POKER AI SYSTEM                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Card        │  │  Action      │  │  CFR         │     │
│  │  Abstraction │  │  Abstraction │  │  Agent       │     │
│  │  (K-Means)   │  │  (5 actions) │  │  (Strategy)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            │                                │
│                  ┌─────────▼─────────┐                      │
│                  │  Holdem HU Env    │                      │
│                  │  (Game Rules)     │                      │
│                  └─────────┬─────────┘                      │
│                            │                                │
│                  ┌─────────▼─────────┐                      │
│                  │   GUI / CLI       │                      │
│                  │   (Interface)     │                      │
│                  └───────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. **Card Abstraction** (`src/poker/abstraction/card_abstraction.py`)

**Purpose**: Reduce 2.6 million unique poker hands to manageable buckets.

**How it works**:

#### Preflop (169 buckets - LOSSLESS)
```python
# Example: Pocket Aces
hole_cards = ['As', 'Ah']
bucket = get_preflop_bucket(hole_cards)  # Returns bucket 1

# Logic:
# - Suits don't matter preflop
# - Only ranks + suited/unsuited matter
# - AA, KK, QQ... = buckets 1-13 (pairs)
# - AK, AQ, AJ... = buckets 14-91 (unsuited)
# - AKs, AQs... = buckets 92-169 (suited)
```

#### Postflop (50 flop, 50 turn, 10 river buckets - EQUITY-BASED)
```python
# Example: AsKh on Qs9s3c flop
hole = ['As', 'Kh']
board = ['Qs', '9s', '3c']

# Step 1: Calculate equity distribution (20 MC samples)
equities = []
for _ in range(20):
    random_opponent = deal_random_hand()
    random_runout = complete_board()
    if I_win(my_hand, opp_hand, final_board):
        equities.append(1.0)
    else:
        equities.append(0.0)

# Step 2: Create histogram
hist = np.histogram(equities, bins=50)  # Shape: (50,)

# Step 3: K-Means prediction
bucket = kmeans_flop.predict([hist])[0]  # Returns 0-49
```

**Why K-Means?**
- Hands with similar equity *distributions* cluster together
- AA vs KKK (strong made hand) → high equity, tight distribution
- AsKs on Qs9s3c (flush draw) → bimodal distribution (hit or miss)
- These get different buckets despite similar average equity!

---

### 2. **Action Abstraction** (`src/poker/abstraction/action_abstraction.py`)

**Purpose**: Reduce 9 environment actions to 5 discrete actions.

**Mapping**:
```
Environment (9 actions)          →  Abstract (5 actions)
├─ 0: Fold                       →  0: FOLD
├─ 1: Check/Call                 →  1: CHECK/CALL
├─ 2: Min-raise                  →  2: BET_SMALL (1/3 pot)
├─ 3: 0.25 pot                   →  2: BET_SMALL
├─ 4: 0.33 pot                   →  2: BET_SMALL
├─ 5: 0.5 pot                    →  3: BET_LARGE (pot-size)
├─ 6: 0.75 pot                   →  3: BET_LARGE
├─ 7: Pot-size                   →  4: BET_LARGE
└─ 8: All-in                     →  4: BET_LARGE
```

**Why abstract?**
- Players don't really distinguish between 0.73 pot and 0.75 pot raises
- Reduces strategy complexity by 80% (5 vs 9 actions)
- CFR converges 5x faster

---

### 3. **CFR Agent** (`src/poker/agents/cfr_agent.py`)

**The Brain** - This is where the magic happens.

#### Key Data Structures:
```python
class CFRAgent:
    def __init__(self):
        # Core strategy tables (hashmap of info_set → array)
        self.regret_sum = defaultdict(lambda: np.zeros(5))     # Cumulative regrets
        self.strategy_sum = defaultdict(lambda: np.zeros(5))   # Cumulative strategy
        
        # Example entry after training:
        # regret_sum["1|flop|20|10"] = [0.0, -5.2, 8.3, 12.1, -3.4]
        #                                 ^    ^     ^    ^     ^
        #                               fold call  small large allin
        
        # strategy_sum["1|flop|20|10"] = [0.0, 0.3, 0.5, 0.2, 0.0]
        #                                      → 30% call, 50% small bet, 20% large bet
```

#### Information Set Format:
```python
info_set = f"{bucket}|{street}|{pot}|{to_call}"

# Example: "42|turn|150|50"
# Means:
# - Bucket 42 (specific hand strength range)
# - Turn street
# - Pot is 150 BB
# - Need to call 50 BB to continue

# Why this format?
# - Captures all decision-relevant information
# - Different pots/bet-sizes = different optimal strategies
# - Same hand on turn vs river = different strategies
```

---

### 4. **The CFR Algorithm** - Step by Step

#### Training Loop (External Sampling CFR):

```python
def train_iteration(env, player_idx):
    """
    One iteration of CFR for player_idx.
    
    This is called 25,000 times (12,500 for P0, 12,500 for P1)
    """
    env.reset()  # New random hand
    return _external_sampling_cfr(env, player_idx)
```

#### Recursive Tree Traversal:

```python
def _external_sampling_cfr(env, player_idx):
    """
    CORE ALGORITHM - Read this carefully!
    
    Recursively traverses the game tree, computing:
    1. Counterfactual values for each action
    2. Regrets (how much better each action would have been)
    3. Strategy adjustments based on regrets
    """
    
    # BASE CASE: Game over
    if env.done:
        return env.rewards[player_idx]  # +1.5 BB (won) or -1.0 BB (lost)
    
    # Get current game state
    current_player = env.current_player
    legal_actions = env._get_legal_actions()  # [0, 1, 2, 3, 4]
    
    # Create info set string
    info_set = self._get_info_set(env, current_player)
    # Example: "42|turn|150|50"
    
    # Get current strategy (regret matching)
    strategy = self.get_strategy(info_set, legal_actions)
    # Example: [0.0, 0.3, 0.5, 0.2, 0.0]
    
    # CASE 1: Updating player (traverser)
    if current_player == player_idx:
        # Save environment state
        saved_state = self._save_env_state(env)
        
        # Try EVERY legal action, compute utility
        action_utilities = np.zeros(5)
        for abstract_action in [1, 2, 3, 4]:  # fold, call, small, large, allin
            # Restore state
            self._restore_env_state(env, saved_state)
            
            # Map abstract → env action
            env_action = abstract_to_env[abstract_action]  # 1→1, 2→4, 3→6, 4→7
            
            # Take action
            env.step(env_action)
            
            # RECURSE: What's the value of this action?
            action_utilities[abstract_action] = self._external_sampling_cfr(env, player_idx)
        
        # Expected utility (how good is this situation on average?)
        utility = np.sum(strategy * action_utilities)
        # Example: 0.3*(-2.1) + 0.5*3.4 + 0.2*5.2 = 2.14 BB
        
        # Compute regrets (how much better was each action?)
        regrets = action_utilities - utility
        # Example: 
        # - Calling: -2.1 - 2.14 = -4.24 (bad!)
        # - Small bet: 3.4 - 2.14 = +1.26 (good!)
        # - Large bet: 5.2 - 2.14 = +3.06 (even better!)
        
        # Update cumulative regrets
        self.regret_sum[info_set] += regrets
        
        # Update cumulative strategy
        self.strategy_sum[info_set] += strategy
        
        # Restore and return
        self._restore_env_state(env, saved_state)
        return utility
        
    # CASE 2: Opponent (sample one action)
    else:
        # Sample action according to current strategy
        sampled_action = np.random.choice(5, p=strategy)
        
        # Map and execute
        env_action = abstract_to_env[sampled_action]
        env.step(env_action)
        
        # RECURSE
        return self._external_sampling_cfr(env, player_idx)
```

#### Regret Matching (Strategy Update):

```python
def get_strategy(info_set, legal_actions):
    """
    Convert regrets → probabilities using regret matching.
    
    Key insight: Play actions proportional to their positive regrets!
    """
    regrets = self.regret_sum[info_set]
    # Example: [0, -4.2, 8.3, 12.1, -3.4]
    
    # Only keep positive regrets
    positive_regrets = np.maximum(regrets, 0)
    # → [0, 0, 8.3, 12.1, 0]
    
    # Normalize to probabilities
    total = np.sum(positive_regrets)
    if total > 0:
        strategy = positive_regrets / total
        # → [0, 0, 0.41, 0.59, 0]
        # Meaning: 41% small bet, 59% large bet
    else:
        # No regrets yet → uniform random
        strategy = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    return strategy
```

---

### 5. **Holdem HU Environment** (`src/poker/envs/holdem_hu_env.py`)

**The Game Engine** - Enforces poker rules.

#### Key Methods:

```python
class HoldemHuEnv:
    def reset(self):
        """Start new hand"""
        self.deck.reset()
        self.hands = [deck.draw(2), deck.draw(2)]  # Deal hole cards
        self.board = []
        self.pot = SB + BB  # 0.5 + 1.0 = 1.5 BB
        self.street = PREFLOP
        self.current_player = SB_idx  # Small blind acts first preflop
        
    def step(self, action_id):
        """Execute action, return (obs, reward, done, truncated, info)"""
        
        # Validate action
        if action_id not in legal_actions:
            return obs, -100, True, False, {"illegal": True}  # Huge penalty
        
        # Execute
        reward, done = self._execute_action(action_id)
        
        # If not done, switch player
        if not done:
            self.current_player = 1 - self.current_player
        
        return self._get_obs(), reward, done, False, {}
    
    def _execute_action(self, action_id):
        """Core game logic"""
        
        if action_id == FOLD:
            # Opponent wins pot
            winner = 1 - self.current_player
            self.stacks[winner] += self.pot
            return self._calc_rewards(), True
            
        elif action_id == CHECK_CALL:
            # Match opponent's bet
            to_call = opp_investment - my_investment
            self.stacks[current_player] -= to_call
            self.pot += to_call
            
            # If betting complete, advance street
            if self._betting_complete():
                if self.street == RIVER:
                    return self._showdown()  # Compare hands
                else:
                    self._deal_next_street()
                    return 0, False
            return 0, False
            
        else:  # RAISE
            # Calculate raise size
            if action_id == MIN_RAISE:
                raise_to = opp_investment + BB
            elif action_id == RAISE_HALF_POT:
                raise_to = opp_investment + pot * 0.5
            # ... etc
            
            # Put chips in
            amount = min(raise_to - my_investment, my_stack)
            self.stacks[current_player] -= amount
            self.pot += amount
            
            # Check all-in
            if my_stack == 0 or opp_stack == 0:
                return self._runout_and_showdown()
                
            return 0, False
    
    def _showdown(self):
        """Evaluate hands at showdown"""
        score0 = HandEvaluator.evaluate(hands[0] + board)  # Lower = better
        score1 = HandEvaluator.evaluate(hands[1] + board)
        
        if score0 < score1:
            winner = 0
        elif score1 < score0:
            winner = 1
        else:
            # Tie - split pot
            self.stacks[0] += pot / 2
            self.stacks[1] += pot / 2
            return self._calc_rewards(), True
        
        self.stacks[winner] += pot
        return self._calc_rewards(), True
```

---

## Data Flow: From Hand to Decision

### Example Hand Walkthrough:

```
SITUATION:
You have: As Kh
Board: Qs 9s 3c (flop)
Pot: 20 BB
Opponent bets 10 BB
Your stack: 90 BB

DECISION PROCESS:

1. GUI receives click on "Call" button
   └─> Calls human_action(action_id=1)

2. GUI validates and executes:
   env.step(1)

3. Environment processes call:
   - Deducts 10 BB from your stack
   - Adds 10 BB to pot (now 30 BB)
   - Betting round complete → advance to turn

4. Turn card dealt: 2h
   Board now: Qs 9s 3c 2h

5. Bot's turn to act:
   
   a) Create info set:
      hole = ['4d', '3s']  # Bot's hidden cards
      board_str = ['Qs', '9s', '3c', '2h']
      
   b) Get bucket:
      equity_dist = calc_equity_distribution(hole, board, n_samples=20)
      bucket = kmeans_turn.predict([equity_dist])[0]
      # Example: bucket = 23
      
   c) Build info set string:
      info_set = f"{23}|turn|{30}|{0}"
      # "23|turn|30|0" (bucket 23, turn, 30 BB pot, 0 to call)
      
   d) Get strategy from trained model:
      avg_strategy = agent.get_average_strategy(info_set, legal_actions)
      # Example: [0, 0.2, 0.5, 0.3, 0]
      # → 20% check, 50% small bet, 30% large bet
      
   e) Sample action:
      abstract_action = np.random.choice(5, p=avg_strategy)
      # Example: samples 2 (small bet)
      
   f) Map to environment:
      env_action = abstract_to_env[2]  # 2 → 4 (1/3 pot raise)
      
   g) Execute:
      env.step(4)
      # Bot bets 10 BB (1/3 of 30 BB pot)

6. GUI updates display:
   - Shows new bet
   - Enables/disables buttons
   - Waits for your response
```

---

## GUI Architecture (`gui/cfr_poker_gui.py`)

### Component Breakdown:

```python
class CFRPokerGUI:
    def __init__(self):
        # 1. Initialize game components
        self.env = HoldemHuEnv()
        self.card_abstraction = CardAbstraction()
        self.card_abstraction.load_kmeans_models(...)  # Load clusters
        self.agent = CFRAgent(card_abstraction=..., num_actions=5)
        self.agent.load("checkpoints/cfr_kmeans/cfr_abstracted_final.pkl")
        
        # 2. Setup UI
        self.setup_ui()  # Creates Tkinter widgets
        
        # 3. Start first hand
        self.new_hand()
    
    def setup_ui(self):
        """Create all UI elements"""
        # Game info (board, pot, street)
        # Player frames (hand, stack, role)
        # Action history (scrolling text)
        # Action buttons (fold, call, raise...)
        # Session stats (BB won/lost, hands played)
    
    def new_hand(self):
        """Start new hand"""
        self.env.reset()
        self.update_display()
        
        # If bot acts first (button = BB), trigger bot action
        if self.env.current_player == self.bot_idx:
            self.root.after(500, self.bot_action)  # 500ms delay for UI
    
    def human_action(self, action_id):
        """Player clicks button"""
        # 1. Log action to history
        self.history_text.insert(END, f"You: {action_name}\n")
        
        # 2. Execute in environment
        obs, reward, done, truncated, info = self.env.step(action_id)
        
        # 3. Update display
        self.update_display()
        
        # 4. If not done, trigger bot's turn
        if not done:
            self.root.after(500, self.bot_action)
        else:
            self.end_hand()  # Update stats, show winner
    
    def bot_action(self):
        """Bot's decision logic"""
        # 1. Get info set
        info_set = self.agent._get_info_set(self.env, self.bot_idx)
        
        # 2. Get legal actions
        legal_actions = self.env._get_legal_actions()
        
        # 3. Get average strategy (trained policy)
        avg_strat = self.agent.get_average_strategy(info_set, legal_actions)
        
        # 4. Sample action
        abstract_action = np.random.choice(5, p=avg_strat)
        
        # 5. Map to environment action
        env_action = abstract_to_env[abstract_action]
        
        # 6. Execute
        self.env.step(env_action)
        
        # 7. Update display
        self.update_display()
        
        # 8. Check if hand continues
        if not self.env.done:
            # If still bot's turn, recurse
            if self.env.current_player == self.bot_idx:
                self.root.after(500, self.bot_action)
        else:
            self.end_hand()
    
    def end_hand(self):
        """Hand finished - update session stats"""
        rewards = self.env.rewards  # [+2.5, -2.5] for example
        
        # Update cumulative totals
        self.session_stats['player_total_bb'] += rewards[self.human_idx]
        self.session_stats['bot_total_bb'] += rewards[self.bot_idx]
        self.session_stats['hands_played'] += 1
        
        # Update UI labels
        self.player_total_label.config(text=f"{player_total:+.1f} BB")
        self.bot_total_label.config(text=f"{bot_total:+.1f} BB")
```

### Tkinter Event Loop:

```
┌─────────────────────────────────────────┐
│ Main Thread (Tkinter event loop)       │
│                                         │
│  while running:                         │
│    event = get_next_event()             │
│    if event == "Button Click":          │
│      ├─> human_action(action_id)        │
│      └─> ...processing...               │
│                                         │
│    if event == "Timer (after 500ms)":   │
│      ├─> bot_action()                   │
│      └─> ...processing...               │
│                                         │
│    update_display()                     │
└─────────────────────────────────────────┘
```

---

## File Structure

```
turnsports-poker-ai/
├── src/poker/
│   ├── core/
│   │   ├── cards.py          # Card class (rank, suit)
│   │   ├── deck.py           # Deck (shuffle, draw)
│   │   └── hand_eval.py      # Hand strength evaluator
│   │
│   ├── envs/
│   │   └── holdem_hu_env.py  # Game environment (rules, state)
│   │
│   ├── agents/
│   │   ├── cfr_agent.py      # CFR training & playing
│   │   └── llm_agent.py      # LLM-controlled agent
│   │
│   └── abstraction/
│       ├── card_abstraction.py    # K-Means bucketing
│       └── action_abstraction.py  # 9→5 action reduction
│
├── data/kmeans/
│   ├── kmeans_flop_latest.pkl    # Flop clusters
│   └── kmeans_turn_latest.pkl    # Turn clusters
│
├── checkpoints/cfr_kmeans/
│   ├── cfr_abstracted_5000.pkl   # Checkpoint at 5k iters
│   ├── cfr_abstracted_10000.pkl
│   └── cfr_abstracted_final.pkl  # Final trained model
│
├── gui/
│   └── cfr_poker_gui.py      # Tkinter GUI
│
├── scripts/
│   ├── generate_kmeans_clusters.py  # Generate K-Means models
│   └── train_cfr_kmeans.py          # Train CFR agent
│
└── experiments/holdem/
    └── train_cfr_abstracted.py      # Training logic
```

---

## Summary: What Makes This Bot Smart?

### 1. **Meaningful Hand Abstraction**
- Uses equity distributions, NOT random hashing
- AA and 72o are in different buckets (not same like before)
- Flush draws grouped together based on potential

### 2. **Game-Theoretic Learning (CFR)**
- Learns Nash equilibrium strategy through self-play
- Exploitability decreases over iterations
- Not heuristic-based like "if AA then raise"

### 3. **Smart Action Selection**
- Doesn't just pick highest EV action
- Uses *mixed strategies* (randomizes based on learned probabilities)
- Unpredictable to opponents

### 4. **Efficient State Representation**
- Info sets capture: hand strength + pot size + bet amount
- Same hand in different pots → different strategies
- Scales to real-world poker complexity

---

## Next Steps

While training runs (25k iterations = ~2 hours), you can:

1. **Test current 5k bot** in GUI - see how it plays
2. **Read this document** - understand the architecture
3. **Explore code** - trace through a hand execution
4. **Compare bots** - play against 5k vs 25k checkpoint

Once 25k training completes, you'll have a **significantly stronger bot** that:
- Makes better preflop decisions
- Understands pot odds
- Balances bluffs and value bets
- Plays closer to Nash equilibrium

**The bot is currently learning. Give it time to get smarter!** 🧠🃏
