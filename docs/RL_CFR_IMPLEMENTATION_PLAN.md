# RL-CFR Implementation Plan for TurnSports Poker AI

## Executive Summary

This document outlines a comprehensive plan to implement **Reinforcement Learning Counterfactual Regret Minimization (RL-CFR)** for your poker AI project, drawing inspiration from:
1. Your existing Kuhn poker RL-CFR work
2. The [Gongsta/Poker-AI](https://github.com/Gongsta/Poker-AI/) open-source project
3. The Science article on Libratus/DeepStack (poker AI that beat pros)

---

## 1. Understanding the Landscape

### Current State of Your Project

**✅ What You Have:**
- Complete Hold'em environment with 9-action bet abstraction
- CFR agent implementation (External Sampling CFR)
- DQN agent (working but suboptimal for poker - see `CFR_VS_DQN.md`)
- Agentic AI Coach
- GUI infrastructure
- Kuhn poker experience (RL-CFR concepts)

**❌ What's Missing:**
- Robust card abstraction for Hold'em CFR
- Action abstraction aligned with CFR convergence
- RL-CFR hybrid for large game spaces
- Blueprint strategy + real-time solving (Libratus approach)

### Poker-AI Repo Key Learnings

**Architecture Highlights:**
1. **Abstraction Strategy:**
   - Preflop: 169 lossless buckets (card suits don't matter)
   - Flop: 50 clusters (equity distributions + K-Means)
   - Turn: 50 clusters (equity distributions + K-Means)
   - River: 10 clusters (simple equity bucketing)
   - **Total**: ~125,000 information sets (manageable!)

2. **Bet Abstraction:**
   - Limited to 11 betting sequences per street
   - Actions: `k` (check), `bMIN` (1/3 pot), `bMAX` (pot-size), `c` (call), `f` (fold)
   - Prevents infinite raise sequences
   - Dramatically reduces game tree

3. **CFR Implementation:**
   - Uses External Sampling CFR (same as yours!)
   - Separate modules: `preflop_holdem.py` and `postflop_holdem.py`
   - Regret matching with cumulative regret sums
   - Average strategy for Nash approximation

4. **Data Generation:**
   - Pre-compute equity distributions using Monte Carlo
   - K-Means clustering with Earth Mover's Distance
   - Parallel processing for efficiency

---

## 2. RL-CFR: The Hybrid Approach

### What is RL-CFR?

**Classic CFR:**
- Tabular method: stores regrets for every information set
- Works for small games (Kuhn poker: ~12 infosets)
- Doesn't scale to Hold'em (10^160+ infosets)

**RL-CFR:**
- Uses **neural networks** to approximate regret and strategy functions
- Instead of storing `regret_sum[infoset]`, use `regret_network(state) → regrets`
- Enables generalization across similar game states
- Used in AlphaGo, AlphaZero, and modern poker AIs

**Why RL-CFR?**
- Handles Hold'em's massive state space
- Generalizes to unseen situations
- Combines CFR's theoretical guarantees with deep learning's power

### RL-CFR vs Pure CFR

| Aspect | Pure CFR | RL-CFR |
|--------|----------|---------|
| **Storage** | Tables per infoset | Neural network weights |
| **Scalability** | Limited (~1M infosets) | Millions+ infosets |
| **Generalization** | None | Strong |
| **Convergence** | Proven to Nash | Approximate Nash |
| **Speed** | Fast lookup | Slower (NN forward pass) |
| **Best For** | Abstracted games | Full games |

**For Your Project:**
- **Phase 1**: Use **Pure CFR with abstraction** (like Poker-AI repo)
- **Phase 2**: Upgrade to **RL-CFR** for full game

---

## 3. Implementation Phases

### Phase 1: CFR with Smart Abstraction (Week 1-2)

**Goal**: Get a working CFR agent that converges to near-Nash for abstracted Hold'em.

#### Step 1.1: Implement Card Abstraction
```python
# src/poker/abstraction/card_abstraction.py

class CardAbstraction:
    """
    Bucket hands into clusters for CFR.
    """
    
    def __init__(self):
        self.preflop_clusters = 169  # Lossless
        self.flop_clusters = 50      # Equity distribution
        self.turn_clusters = 50      # Equity distribution
        self.river_clusters = 10     # Simple equity
    
    def get_preflop_bucket(self, hole_cards):
        """
        169 lossless buckets:
        - 13 pocket pairs
        - 78 unsuited combinations
        - 78 suited combinations
        """
        # Implementation from Poker-AI repo
        pass
    
    def get_postflop_bucket(self, hole_cards, board, street):
        """
        Equity-based bucketing:
        1. Calculate equity distribution (histogram)
        2. Use pre-trained K-Means to assign cluster
        """
        equity_dist = self._calc_equity_distribution(hole_cards, board)
        
        if street == 'flop':
            return self.flop_kmeans.predict([equity_dist])[0]
        elif street == 'turn':
            return self.turn_kmeans.predict([equity_dist])[0]
        else:  # river
            equity = self._calc_simple_equity(hole_cards, board)
            return min(9, int(equity * 10))  # 10 buckets
    
    def _calc_equity_distribution(self, hole_cards, board, n_samples=1000, n_bins=50):
        """
        Monte Carlo equity distribution:
        1. Sample opponent hands
        2. Roll out remaining board cards
        3. Compute win probability distribution
        """
        # Histogram of equity across different runouts
        pass
```

**Key Insight**: Poker-AI shows that 50x50x10 = 25,000 clusters is enough!

#### Step 1.2: Implement Action Abstraction
```python
# src/poker/abstraction/action_abstraction.py

class ActionAbstraction:
    """
    Simplified bet sizes for CFR training.
    """
    
    DISCRETE_ACTIONS = ['fold', 'check', 'call', 'bMIN', 'bMAX']
    
    def __init__(self):
        self.min_bet_frac = 1/3  # 1/3 pot
        self.max_bet_frac = 1.0  # Pot-size
    
    def abstract_action(self, raw_action, pot, legal_actions):
        """
        Map continuous bet sizes to discrete actions.
        """
        if 'raise_0.25' in raw_action or 'raise_0.33' in raw_action:
            return 'bMIN'
        elif 'raise_0.75' in raw_action or 'raise_1.0' in raw_action:
            return 'bMAX'
        # etc.
    
    def get_legal_actions(self, history):
        """
        Limit to 11 betting sequences (like Poker-AI):
        - kk, kbMINf, kbMINc, kbMAXf, kbMAXc
        - bMINf, bMINc, bMINbMAXf, bMINbMAXc
        - bMAXf, bMAXc
        """
        pass
```

**Key Insight**: Limiting bet sequences prevents infinite game trees.

#### Step 1.3: CFR Training with Abstraction
```python
# experiments/holdem/train_cfr_abstracted.py

from src.poker.agents.cfr_agent import CFRAgent
from src.poker.abstraction.card_abstraction import CardAbstraction
from src.poker.abstraction.action_abstraction import ActionAbstraction

def train_cfr_abstracted(n_iterations=100_000):
    """
    Train CFR on abstracted Hold'em.
    """
    card_abstraction = CardAbstraction()
    action_abstraction = ActionAbstraction()
    cfr_agent = CFRAgent(
        card_abstraction=card_abstraction,
        action_abstraction=action_abstraction
    )
    
    for i in tqdm(range(n_iterations)):
        # Sample random cards
        env = create_abstracted_env()
        
        # Run CFR iteration for both players
        for player_idx in [0, 1]:
            cfr_agent.train_iteration(env, player_idx)
        
        if i % 10_000 == 0:
            # Compute exploitability
            exploitability = compute_exploitability(cfr_agent)
            print(f"Iteration {i}: Exploitability = {exploitability:.4f} BB/hand")
            
            # Save checkpoint
            cfr_agent.save(f'checkpoints/cfr_iter_{i}.pkl')
```

**Expected Results:**
- After 10,000 iterations: Exploitability < 50 mBB/hand
- After 100,000 iterations: Exploitability < 10 mBB/hand
- Near-Nash equilibrium for abstracted game

---

### Phase 2: RL-CFR with Neural Networks (Week 3-4)

**Goal**: Replace tabular CFR with neural function approximators.

#### Step 2.1: Define RL-CFR Architecture
```python
# src/poker/agents/rlcfr_agent.py

import torch
import torch.nn as nn

class RegretNetwork(nn.Module):
    """
    Neural network that predicts regrets for each action.
    """
    def __init__(self, state_dim=200, hidden_dim=512, num_actions=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, state):
        return self.network(state)

class StrategyNetwork(nn.Module):
    """
    Neural network that outputs action probabilities.
    """
    def __init__(self, state_dim=200, hidden_dim=512, num_actions=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class RLCFRAgent:
    """
    CFR with neural network function approximation.
    """
    def __init__(self, state_dim=200, num_actions=5):
        self.regret_net = RegretNetwork(state_dim, num_actions)
        self.strategy_net = StrategyNetwork(state_dim, num_actions)
        
        self.regret_optimizer = torch.optim.Adam(self.regret_net.parameters(), lr=1e-3)
        self.strategy_optimizer = torch.optim.Adam(self.strategy_net.parameters(), lr=1e-3)
        
        self.regret_memory = []  # Store (state, regrets) pairs
        self.strategy_memory = []  # Store (state, strategy) pairs
    
    def get_strategy(self, state, legal_actions):
        """
        Regret matching using neural network predictions.
        """
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            regrets = self.regret_net(state_tensor).numpy()
        
        # Regret matching
        positive_regrets = np.maximum(regrets, 0)
        positive_regrets[~legal_actions] = 0  # Mask illegal actions
        
        regret_sum = positive_regrets.sum()
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            strategy = np.ones_like(regrets) / len(legal_actions)
            strategy[~legal_actions] = 0
        
        return strategy
    
    def train_iteration(self, env, player_idx):
        """
        One iteration of RL-CFR.
        1. Traverse game tree (like vanilla CFR)
        2. Store (state, regret) and (state, strategy) pairs
        3. Periodically update neural networks
        """
        # Reset environment
        state, info = env.reset()
        
        # Traverse with external sampling
        self._traverse(env, player_idx)
        
        # Update networks every N iterations
        if len(self.regret_memory) > 10000:
            self._update_networks()
    
    def _update_networks(self):
        """
        Supervised learning to fit networks to accumulated data.
        """
        # Sample mini-batch
        batch_indices = np.random.choice(len(self.regret_memory), size=256)
        
        states = torch.FloatTensor([self.regret_memory[i][0] for i in batch_indices])
        target_regrets = torch.FloatTensor([self.regret_memory[i][1] for i in batch_indices])
        target_strategies = torch.FloatTensor([self.strategy_memory[i][1] for i in batch_indices])
        
        # Update regret network (MSE loss)
        pred_regrets = self.regret_net(states)
        regret_loss = nn.MSELoss()(pred_regrets, target_regrets)
        
        self.regret_optimizer.zero_grad()
        regret_loss.backward()
        self.regret_optimizer.step()
        
        # Update strategy network (cross-entropy loss)
        pred_strategies = self.strategy_net(states)
        strategy_loss = nn.KLDivLoss()(torch.log(pred_strategies), target_strategies)
        
        self.strategy_optimizer.zero_grad()
        strategy_loss.backward()
        self.strategy_optimizer.step()
        
        print(f"Regret Loss: {regret_loss.item():.4f}, Strategy Loss: {strategy_loss.item():.4f}")
```

**Key Insight**: Neural networks generalize regret/strategy to unseen states!

#### Step 2.2: RL-CFR Training Loop
```python
# experiments/holdem/train_rlcfr.py

def train_rlcfr(n_iterations=1_000_000):
    """
    Train RL-CFR agent.
    """
    agent = RLCFRAgent(state_dim=200, num_actions=5)
    env = HoldemHUEnv()
    
    for i in tqdm(range(n_iterations)):
        # Alternate between players
        player_idx = i % 2
        
        agent.train_iteration(env, player_idx)
        
        if i % 50_000 == 0:
            # Evaluate
            avg_reward = evaluate_agent(agent, n_hands=10_000)
            print(f"Iteration {i}: Avg reward = {avg_reward:.4f} BB/hand")
            
            # Save
            agent.save(f'checkpoints/rlcfr_{i}.pt')
```

**Expected Results:**
- Converges slower than tabular CFR, but handles full game
- Can play unseen card combinations (generalization)
- Approximate Nash equilibrium

---

### Phase 3: Blueprint Strategy + Real-Time Solving (Week 5-6)

**Goal**: Implement Libratus-style approach (blueprint + refinement).

#### Libratus Architecture
1. **Offline**: Train blueprint strategy with abstracted CFR
2. **Online**: Refine strategy in real-time using depth-limited solving

```python
# src/poker/agents/libratus_agent.py

class LibratusAgent:
    """
    Libratus-style agent:
    1. Blueprint strategy (pre-computed CFR)
    2. Real-time depth-limited solving
    """
    
    def __init__(self, blueprint_path):
        self.blueprint = load_blueprint(blueprint_path)  # Pre-trained CFR
        self.abstraction = CardAbstraction()
    
    def get_action(self, env):
        """
        1. Abstract current state
        2. Look up blueprint action
        3. (Optional) Refine with depth-limited solving
        """
        # Abstract state
        abstract_state = self.abstraction.abstract(env)
        
        # Get blueprint action
        blueprint_strategy = self.blueprint.get_strategy(abstract_state)
        
        # Real-time solving (optional, expensive)
        if self.should_refine(env):
            refined_strategy = self._depth_limited_solve(env, depth=2)
            return refined_strategy
        else:
            return blueprint_strategy
    
    def _depth_limited_solve(self, env, depth):
        """
        Run CFR for next 'depth' streets only.
        Uses current state as root.
        """
        # Mini-CFR from current position
        mini_cfr = CFRAgent()
        
        for _ in range(1000):  # Quick solve
            mini_cfr.train_iteration(env, player_idx=env.current_player)
        
        return mini_cfr.get_average_strategy(env.get_info_set())
```

---

## 4. Recommended Implementation Order

### Week 1-2: Abstracted CFR
1. ✅ Implement card abstraction (preflop 169, postflop equity clustering)
2. ✅ Implement action abstraction (5 actions, 11 sequences)
3. ✅ Adapt your CFR agent to use abstractions
4. ✅ Train on abstracted game (100k iterations)
5. ✅ Evaluate exploitability

**Deliverable**: CFR agent that beats TAG/LAG opponents.

### Week 3-4: RL-CFR
1. ✅ Implement RegretNetwork and StrategyNetwork
2. ✅ Implement RL-CFR training loop
3. ✅ Train on full game (no abstraction needed!)
4. ✅ Compare performance to abstracted CFR

**Deliverable**: RL-CFR agent that generalizes to novel situations.

### Week 5-6: Advanced Features
1. ✅ Implement blueprint + real-time solving
2. ✅ Add hand history analysis
3. ✅ Integrate with AI coach (show GTO vs exploitative advice)
4. ✅ Build web dashboard

**Deliverable**: Production-ready poker AI system.

---

## 5. Integration with Your Existing Project

### File Structure
```
poker-rl-cfr/
├── src/poker/
│   ├── abstraction/
│   │   ├── card_abstraction.py      # NEW
│   │   ├── action_abstraction.py    # NEW
│   │   └── equity_calculator.py     # NEW
│   ├── agents/
│   │   ├── cfr_agent.py             # EXISTING (modify to use abstraction)
│   │   ├── rlcfr_agent.py           # NEW
│   │   ├── kuhn_rlcfr.py            # EXISTING (keep as reference)
│   │   └── libratus_agent.py        # NEW (Phase 3)
│   └── ...
├── experiments/
│   ├── kuhn/
│   │   └── train_kuhn_cfr.py        # EXISTING
│   └── holdem/
│       ├── train_cfr_abstracted.py  # NEW
│       ├── train_rlcfr.py           # NEW
│       └── eval_cfr.py              # NEW
└── ...
```

### Modify Existing CFR Agent
```python
# src/poker/agents/cfr_agent.py - ADD abstraction support

class CFRAgent:
    def __init__(self, card_abstraction=None, action_abstraction=None):
        self.card_abstraction = card_abstraction
        self.action_abstraction = action_abstraction
        # ... existing code ...
    
    def _get_info_set(self, env, player):
        """
        MODIFIED: Use abstraction if available.
        """
        if self.card_abstraction:
            # Get abstract bucket instead of raw cards
            bucket = self.card_abstraction.get_bucket(
                env.hands[player], 
                env.board, 
                env.street
            )
            info_set = f"{bucket}|{env.street}|..."
        else:
            # Original implementation (for Kuhn poker)
            hand_str = ''.join(sorted([str(c) for c in env.hands[player]]))
            info_set = f"{hand_str}|..."
        
        return info_set
```

---

## 6. Key Takeaways from Poker-AI Repo

1. **Abstraction is King**: 169 preflop + 50x50x10 postflop = manageable
2. **Limit Betting Sequences**: 11 sequences prevent infinite trees
3. **External Sampling CFR**: Same as yours - efficient for large games
4. **Parallelization**: Use `joblib` for equity calculations
5. **Modularity**: Separate abstraction, training, and evaluation

---

## 7. Metrics for Success

### CFR (Abstracted)
- **Exploitability**: < 10 mBB/hand after 100k iterations
- **vs Random**: +30 BB/100
- **vs TAG**: +5 BB/100 (near breakeven is GTO)
- **Convergence**: Regret sums stabilize

### RL-CFR
- **Generalization**: Wins against unseen opponent strategies
- **Sample Efficiency**: Fewer iterations than tabular CFR
- **Performance**: Within 5% of tabular CFR on abstracted game

---

## 8. References

1. **Your Kuhn Poker Work**: Foundation for RL-CFR logic
2. **Gongsta/Poker-AI**: Abstraction techniques, code structure
3. **Libratus (Science 2017)**: Blueprint + real-time solving
4. **DeepStack (Science 2017)**: Continual re-solving
5. **Pluribus (Science 2019)**: Multi-player poker CFR

---

## 9. Next Steps

**Immediate (This Week):**
1. Read and understand `abstraction.py` from Poker-AI
2. Implement `CardAbstraction` class
3. Generate preflop buckets (169)
4. Test on simplified environment

**Short-term (Next 2 Weeks):**
1. Implement equity distribution calculations
2. Pre-compute K-Means clusters for postflop
3. Integrate with your CFR agent
4. Train on abstracted Hold'em

**Long-term (Month 2):**
1. Build RL-CFR agent
2. Benchmark against abstracted CFR
3. Add blueprint strategy
4. Deploy to production

---

## 10. Questions to Consider

1. **How much abstraction?** Start aggressive (169/50/50/10), refine later
2. **CFR vs RL-CFR?** CFR first (proven), RL-CFR later (research)
3. **Training time?** Expect 6-12 hours for 100k CFR iterations on CPU
4. **Evaluation?** Measure exploitability, not just win rate
5. **Integration with coach?** Coach can show "GTO play" (CFR) vs "exploit" (DQN)

---

**Let's build this! 🚀**

Your Kuhn poker RL-CFR work gives you a huge head start. The Poker-AI repo shows the path. Now we execute.
