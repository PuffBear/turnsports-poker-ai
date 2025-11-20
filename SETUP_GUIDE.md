# SETUP GUIDE

## Prerequisites

- Python 3.8+
- pip

## Installation

### Step 1: Install Dependencies

```bash
pip install numpy pandas matplotlib torch gymnasium termcolor tqdm
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python scripts/quickstart_test.py
```

Expected output:
```
==============================================================
Quick Start Test - Poker RL Project
==============================================================

Testing Hold'em Environment...
✓ Environment reset successful
  Player 0 hand: [Ah, Kd]
  Player 1 hand: [Qs, Jc]
  Board: []
  Pot: 1.5 BB
✓ Hand completed in 12 actions
  Final board: [Kc, 7s, 2d, 9h, 3s]
  Final pot: 25.0 BB
  Winner: Player 0

✓ Environment test passed!

Testing Agentic Coach...
✓ Coach recommendation: Check/Call
  Tool results: {'equity': 0.523, 'pot_odds': 0.33, ...}

✓ Coach test passed!

==============================================================
All tests passed!
==============================================================
```

## Usage

### Option 1: Quick Play (No Training)

Play against a random bot immediately:

```bash
python gui/holdem_poker_gui.py
```

**Note:** Coach will work (equity + pot odds), but no rollout simulations without trained bot.

### Option 2: Full Pipeline (Train Bot First)

#### 2.1 Train the Bot

```bash
python experiments/holdem/train_holdem_dqn_vs_pool.py
```

**Training time**: 2-6 hours on CPU (50,000 episodes)

**Output**: Checkpoints saved in `checkpoints/` every 5,000 episodes

**Progress**:
```
Training DQN Agent for Heads-Up No-Limit Hold'em
Training for 50000 episodes...

[Episode 5000]
  Avg Reward (last 1000): +2.35 BB
  Avg Episode Length: 15.3 steps
  Avg Loss: 0.0234
  Epsilon: 0.782

... continues ...

Training complete!
Final model saved to checkpoints/dqn_final.pt
```

#### 2.2 Evaluate Performance

```bash
python experiments/holdem/eval_bot_vs_baselines.py
```

**Output**:
```
Evaluating DQN Agent vs Baseline Opponents
Evaluating over 10000 hands per opponent...

RandomOpponent:
  Win Rate:   65.23%
  bb/100:     +25.67

TAGOpponent:
  Win Rate:   52.14%
  bb/100:     +8.34

LAGOpponent:
  Win Rate:   58.91%
  bb/100:     +15.22

Overall bb/100: +16.41
✓ Excellent: Agent is crushing the opponents!
```

#### 2.3 Play with Full Coach

```bash
python gui/holdem_poker_gui.py
```

Now coach will use all tools including rollout simulations!

## GUI Controls

### Game Panel (Left)
- **Board**: Community cards
- **Your Hand**: Your hole cards (blue text)
- **Pot**: Current pot size
- **Stacks**: Your stack vs bot stack
- **Action History**: Log of all actions
- **Action Buttons**: 9 buttons for different actions
  - Grayed out = illegal in current state
  - Click to take action
- **New Hand**: Start fresh hand

### Coach Panel (Right)
- **Recommendation**: Suggested action (e.g., "1/2 Pot Raise")
- **Analysis**: Detailed breakdown including:
  - Equity estimate
  - Pot odds calculation
  - Rollout EV comparisons
  - Strategic considerations
- **Get Coach Advice**: Click to update recommendation
  - Auto-updates on your turn
  - Takes 1-3 seconds (running simulations)

## Training Customization

Edit `experiments/holdem/train_holdem_dqn_vs_pool.py`:

```python
# More episodes = better bot (but longer training)
n_episodes = 100000  # Default: 50000

# Save checkpoints more/less frequently
save_interval = 10000  # Default: 5000

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## Coach Customization

Edit `gui/holdem_poker_gui.py`:

```python
# Initialize coach with different settings
coach = AgenticCoach(
    bot_policy=self.bot_policy,
    use_rollouts=True,   # Set False for faster, less accurate advice
    use_equity=True      # Set False to skip equity calculation
)

# Adjust simulation parameters
recommendation = self.coach.get_recommendation(
    self.env,
    n_rollouts=100,       # Default: 100 (more = slower but accurate)
    n_equity_samples=2000 # Default: 2000 (more = accurate equity)
)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'gymnasium'"

**Solution**: Install dependencies
```bash
pip install gymnasium
```

### "No trained agent found"

**Solution**: Either:
1. Train an agent first: `python experiments/holdem/train_holdem_dqn_vs_pool.py`
2. Or play with random bot (coach still works for equity/pot odds)

### "Coach is slow"

**Solutions**:
1. Reduce rollout count: `n_rollouts=50` (default 100)
2. Reduce equity samples: `n_equity_samples=1000` (default 2000)
3. Disable rollouts: `use_rollouts=False`

### Training is slow

**Solutions**:
1. Use GPU: Set `device='cuda'` (requires PyTorch with CUDA)
2. Reduce episodes: `n_episodes=20000` for faster testing
3. Use fewer opponents: Comment out LAG/TAG in opponent list

## File Structure

```
turnsports-poker-ai/
├── src/poker/          # Core library
│   ├── core/           # Cards, deck, evaluation
│   ├── envs/           # Game environments
│   ├── agents/         # RL agents (DQN)
│   ├── opponents/      # Rule-based opponents
│   ├── coach/          # Agentic coach + tools
│   └── analysis/       # Evaluation utilities
├── experiments/        # Training & evaluation scripts
│   └── holdem/
│       ├── train_holdem_dqn_vs_pool.py
│       └── eval_bot_vs_baselines.py
├── gui/                # Interactive GUI
│   └── holdem_poker_gui.py
├── docs/               # Documentation
│   └── holdem/
│       ├── holdem_design_notes.md
│       └── holdem_agentic_coach.md
├── scripts/            # Utility scripts
│   └── quickstart_test.py
├── checkpoints/        # Saved models (created during training)
├── README.md
├── PROJECT_SUMMARY.md
├── SETUP_GUIDE.md (this file)
└── requirements.txt
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Run quickstart test
3. ✅ Play with random bot to familiarize yourself
4. ⬜ Train bot (optional, takes time)
5. ⬜ Evaluate bot performance
6. ⬜ Customize and extend!

## Advanced: Extending the Project

### Add a New Opponent

```python
# In src/poker/opponents/holdem_rule_based.py

class MyOpponent(HoldemOpponent):
    def get_action(self, env, obs):
        # Your strategy here
        legal = env._get_legal_actions()
        return my_action
```

### Add a New Coach Tool

```python
# In src/poker/coach/

def my_tool(state, ...):
    # Analyze something
    return result

# In agentic_coach.py
tool_results['my_tool'] = my_tool(summary, ...)
```

### Switch to Different RL Algorithm

Replace DQN with PPO, A3C, etc. by:
1. Implement new agent in `src/poker/agents/`
2. Update training script to use new agent
3. Ensure policy wrapper works with new agent

## Tips for Best Results

1. **Training**:
   - Train for at least 50k episodes
   - Monitor avg reward - should increase over time
   - Save checkpoints regularly (power outages happen!)

2. **Evaluation**:
   - Use at least 10k hands for statistical significance
   - Compare against multiple opponent types
   - bb/100 > 5 is excellent

3. **Playing**:
   - Pay attention to coach's equity analysis
   - Compare recommended action to your intuition
   - Learn why certain actions have higher EV

4. **Studying**:
   - Review action history after hands
   - Ask "Why did coach recommend differently?"
   - Experiment with different bet sizes

## Resources

- **Poker Strategy**: pokerstrategy.com, upswingpoker.com
- **RL Theory**: Sutton & Barto - "Reinforcement Learning"
- **Game Theory**: "The Mathematics of Poker" by Chen & Ankenman
- **CFR**: "Regret Minimization in Games with Incomplete Information" (Zinkevich et al.)

## Support

Found a bug or have questions?
- Check `PROJECT_SUMMARY.md` for overview
- Read `docs/holdem/holdem_design_notes.md` for architecture
- Inspect code - it's well-commented!

---

**Good luck at the tables! 🃏🎉**
