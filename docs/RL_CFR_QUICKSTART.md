# RL-CFR Quick Start Guide

This guide shows you how to use the newly implemented RL-CFR modules for your poker AI.

## 🎯 What We Built

1. **Card Abstraction** (`src/poker/abstraction/card_abstraction.py`)
   - Preflop: 169 lossless buckets ✅ TESTED
   - Postflop: Equity-based bucketing with K-Means

2. **Action Abstraction** (`src/poker/abstraction/action_abstraction.py`)
   - 5 discrete actions (fold, check, call, bet_small, bet_large)
   - 11 valid betting sequences

3. **Abstracted CFR Agent** (`src/poker/agents/cfr_agent.py`)
   - Modified to support card/action abstraction
   - Reduces state space from 10^160 to ~125k

4. **Training Scripts**
   - Kuhn poker RL-CFR: `experiments/kuhn/train_kuhn_rlcfr.py`
   - Hold'em abstracted CFR: `experiments/holdem/train_cfr_abstracted.py`
   - K-Means generation: `scripts/generate_kmeans_clusters.py`

---

## 🚀 Quick Start (3 Options)

### Option 1: Train CFR with Simple Equity Bucketing (FASTEST)

This uses fast equity bucketing (no K-Means needed).

```bash
cd /Users/Agriya/Desktop/turnsports.ai/turnsports-poker-ai

# Quick test (10k iterations, ~1 hour)
python experiments/holdem/train_cfr_abstracted.py

# Or edit the script to use quick test:
# Uncomment the "# Quick test" section in main()
```

**Expected Results:**
- 10k iterations: ~1 hour
- Memory: ~10 MB
- Info sets: ~5-10k
- Should learn basic strategies

### Option 2: Train CFR with K-Means Clustering (BEST)

This uses sophisticated equity distributions for better abstraction.

**Step 1: Generate K-Means models (once)**
```bash
# This takes 30-60 minutes but only needs to be done once
python scripts/generate_kmeans_clusters.py
```

This creates:
- `data/kmeans/kmeans_flop_latest.pkl`
- `data/kmeans/kmeans_turn_latest.pkl`

**Step 2: Load K-Means in your code**
```python
from src.poker.abstraction import CardAbstraction

abstraction = CardAbstraction()
abstraction.load_kmeans_models(
    'data/kmeans/kmeans_flop_latest.pkl',
    'data/kmeans/kmeans_turn_latest.pkl'
)

# Now use in CFR training...
```

**Step 3: Train CFR**
```bash
python experiments/holdem/train_cfr_abstracted.py
```

**Expected Results:**
- 100k iterations: ~6-12 hours
- Memory: ~50 MB
- Info sets: ~50-100k
- Exploitability < 10 mBB/hand
- Beats TAG/LAG opponents

### Option 3: Train RL-CFR on Kuhn Poker (REFERENCE)

Learn the RL-CFR concept on a simple game first.

```bash
python experiments/kuhn/train_kuhn_rlcfr.py
```

**Expected Results:**
- 100k iterations: ~30-60 minutes
- Converges to Nash equilibrium
- Near-zero exploitability

---

## 📊 Understanding the Output

### Training Progress
```
[Iteration 10,000] (345s, 29.0 iter/s)
  Avg Utility P0: +0.0234
  Avg Utility P1: -0.0198
  Exploitability: 0.0432
  Unique info sets: 8,234
```

- **Utilities**: Should hover around 0 (zero-sum game)
- **Exploitability**: Should decrease over time → 0
- **Info sets**: Should grow then plateau

### Training Curves
After training, check `checkpoints/cfr_abstracted/training_curves.png`:

1. **Utilities**: Should oscillate around 0
2. **Info Set Growth**: Should plateau (all important states visited)
3. **Utility Distribution**: Should be centered at 0
4. **Exploitability**: Should decrease monotonically

---

## 🎮 Using the Trained Agent

### Load and Play
```python
from src.poker.agents.cfr_agent import CFRAgent
from src.poker.abstraction import CardAbstraction

# Load abstraction
abstraction = CardAbstraction()

# Load trained agent
agent = CFRAgent(card_abstraction=abstraction)
agent.load('checkpoints/cfr_abstracted/cfr_abstracted_final.pkl')

# Get action
env.reset()
info_set = agent._get_info_set(env, player_idx=0)
legal_actions = env._get_legal_actions()
strategy = agent.get_average_strategy(info_set, legal_actions)
action = np.random.choice(len(strategy), p=strategy)
```

### Integrate with GUI
```python
# In gui/holdem_poker_gui.py

# Initialize CFR bot
from src.poker.agents.cfr_agent import CFRAgent
from src.poker.abstraction import CardAbstraction

abstraction = CardAbstraction()
cfr_bot = CFRAgent(card_abstraction=abstraction)
cfr_bot.load('checkpoints/cfr_abstracted/cfr_abstracted_final.pkl')

# In bot's turn:
info_set = cfr_bot._get_info_set(env, player_idx=1)
legal_actions = env._get_legal_actions()
strategy = cfr_bot.get_average_strategy(info_set, legal_actions)
bot_action = np.random.choice(len(strategy), p=strategy)
```

---

## 🔍 Testing the Abstractions

### Test Card Abstraction
```bash
python src/poker/abstraction/card_abstraction.py
```

**Expected Output:**
```
Testing Preflop Abstraction:
--------------------------------------------------
✅ Pocket Aces: ['As', 'Ad'] -> Bucket 1 (expected 1)
✅ Pocket Kings: ['Ks', 'Kd'] -> Bucket 2 (expected 2)
✅ AK unsuited: ['As', 'Kd'] -> Bucket 14 (expected 14)
✅ AK suited: ['As', 'Ks'] -> Bucket 92 (expected 92)
```

### Test Action Abstraction
```bash
python src/poker/abstraction/action_abstraction.py
```

**Expected Output:**
```
Testing Action Abstraction:
------------------------------------------------------------
✅ Fold: Action 0 -> f
✅ Check (no bet): Action 1 -> k
✅ 0.25 pot -> BET_MIN: Action 3 -> bMIN
```

---

## 📁 File Structure

```
turnsports-poker-ai/
├── src/poker/
│   ├── abstraction/
│   │   ├── card_abstraction.py    # 169 preflop + equity bucketing
│   │   └── action_abstraction.py  # 5 actions, 11 sequences
│   └── agents/
│       ├── cfr_agent.py           # Modified for abstraction
│       └── kuhn_rlcfr.py          # RL-CFR reference
├── experiments/
│   ├── kuhn/
│   │   └── train_kuhn_rlcfr.py    # Kuhn poker RL-CFR
│   └── holdem/
│       └── train_cfr_abstracted.py # Hold'em abstracted CFR
├── scripts/
│   └── generate_kmeans_clusters.py # K-Means generation
├── data/
│   └── kmeans/                     # K-Means models (generated)
├── checkpoints/
│   └── cfr_abstracted/             # Trained CFR models
└── docs/
    ├── RL_CFR_IMPLEMENTATION_PLAN.md
    └── RL_CFR_SUMMARY.md
```

---

## ⚙️ Configuration Options

### Abstraction Settings

**Card Abstraction:**
```python
abstraction = CardAbstraction()
abstraction.preflop_clusters = 169  # Fixed (lossless)
abstraction.flop_clusters = 50      # Adjustable
abstraction.turn_clusters = 50      # Adjustable
abstraction.river_clusters = 10     # Adjustable
```

**Action Abstraction:**
```python
action_abstraction = ActionAbstraction()
action_abstraction.min_bet_frac = 1/3  # Small bet size
action_abstraction.max_bet_frac = 1.0  # Large bet size (pot)
```

### Training Settings

**CFR Training:**
```python
train_abstracted_cfr(
    n_iterations=100000,      # More = better convergence
    save_interval=10000,      # Checkpoint frequency
    eval_interval=1000,       # Evaluation frequency
    use_action_abstraction=False  # True = 5 actions, False = 9 actions
)
```

**K-Means Generation:**
```python
n_samples = 10000    # More = better clustering (but slower)
n_bins = 50          # Equity histogram granularity
n_clusters_flop = 50 # Number of flop buckets
n_clusters_turn = 50 # Number of turn buckets
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution:**
```bash
# Make sure you're in the project root
cd /Users/Agriya/Desktop/turnsports.ai/turnsports-poker-ai

# Or add to PYTHONPATH
export PYTHONPATH="/Users/Agriya/Desktop/turnsports.ai/turnsports-poker-ai:$PYTHONPATH"
```

### Issue: CFR training very slow
**Solutions:**
1. Reduce `n_iterations` for testing (10k instead of 100k)
2. Reduce abstraction granularity (flop/turn clusters: 50 → 25)
3. Use simple equity bucketing instead of K-Means

### Issue: High memory usage
**Solutions:**
1. Reduce number of clusters
2. Save checkpoints more frequently (clear old regret tables)
3. Use action abstraction (5 actions instead of 9)

### Issue: Postflop abstraction fails
**Cause:** Missing card evaluation module
**Solution:** The equity calculation needs the `HandEvaluator` class. Make sure it's implemented.

---

## 📈 Performance Benchmarks

### DQN vs CFR (Expected)

| Metric | DQN | CFR (Abstracted) |
|--------|-----|------------------|
| vs Random | -10 BB/100 | +30 BB/100 |
| vs TAG | -50 BB/100 | +5 BB/100 |
| vs LAG | -40 BB/100 | +10 BB/100 |
| Exploitability | Unknown | <10 mBB/hand |
| Training Time | 2 hours | 12 hours |

### CFR Convergence Milestones

- **1k iterations**: Random play
- **10k iterations**: Learns fold equity, basic aggression
- **50k iterations**: Near-GTO preflop, decent postflop
- **100k iterations**: Strong overall strategy, <10 mBB exploitability

---

## 🎓 Next Steps

### Week 1: Get CFR Working
1. ✅ Test abstractions (already working!)
2. ⏳ Train quick CFR (10k iterations)
3. ⏳ Verify convergence
4. ⏳ Compare to DQN

### Week 2: Full Training
1. ⏳ Generate K-Means models (optional)
2. ⏳ Train CFR for 100k iterations
3. ⏳ Evaluate against baselines
4. ⏳ Integrate with GUI

### Week 3: Advanced Features
1. ⏳ Implement blueprint + real-time solving (Libratus)
2. ⏳ Add hand history analysis
3. ⏳ Build evaluation dashboard
4. ⏳ Write final documentation

---

## 💡 Pro Tips

1. **Start Simple**: Use simple equity bucketing first, add K-Means later
2. **Monitor Info Sets**: Should plateau around 50-100k
3. **Check Convergence**: Utilities should stabilize, exploitability should decrease
4. **Save Often**: Checkpoints let you resume if training crashes
5. **Visualize**: Training curves reveal convergence issues

---

## 📚 Additional Resources

- **Documentation**: `docs/RL_CFR_IMPLEMENTATION_PLAN.md` (detailed 3-phase plan)
- **Summary**: `docs/RL_CFR_SUMMARY.md` (complete reference)
- **Poker-AI Repo**: https://github.com/Gongsta/Poker-AI/
- **Academic Papers**: See references in implementation plan

---

## ✅ Success Criteria

You'll know it's working when:
1. ✅ Preflop abstraction test passes (already done!)
2. ⏳ CFR training completes without errors
3. ⏳ Info sets grow to 50-100k then plateau
4. ⏳ Exploitability decreases over time
5. ⏳ CFR bot beats DQN bot decisively

---

**Ready to train? Start with:**
```bash
python experiments/holdem/train_cfr_abstracted.py
```

**Questions? Check:**
- `docs/RL_CFR_SUMMARY.md` for detailed explanations
- `docs/RL_CFR_IMPLEMENTATION_PLAN.md` for the full roadmap

**Let's build a poker AI that actually plays poker! 🎰🚀**
