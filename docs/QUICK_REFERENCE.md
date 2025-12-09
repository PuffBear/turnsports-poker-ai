# 🃏 Quick Reference Card

## Installation (One-time)
```bash
pip install numpy pandas matplotlib torch gymnasium termcolor tqdm
```

## Commands

| Task | Command |
|------|---------|
| **Test setup** | `python scripts/quickstart_test.py` |
| **Train bot** | `python experiments/holdem/train_holdem_dqn_vs_pool.py` |
| **Evaluate bot** | `python experiments/holdem/eval_bot_vs_baselines.py` |
| **Play poker** | `python gui/holdem_poker_gui.py` |

## Poker Actions

| Button | Action | When to Use |
|--------|--------|-------------|
| **Fold** | Give up | Weak hand, bad odds |
| **Check/Call** | Match bet | Drawing, medium strength |
| **Min Raise** | Smallest raise | Probe, value thin |
| **1/4 Pot** | Small raise | Bluff, weak value |
| **1/3 Pot** | Small raise | Bluff, protection |
| **1/2 Pot** | Medium raise | Value, semi-bluff |
| **3/4 Pot** | Large raise | Strong value |
| **Pot** | Large raise | Nuts, polarized |
| **All-In** | Entire stack | Commitment, bluff |

## Coach Output Explained

```
Recommended: 1/2 Pot Raise

• Your equity: ~68%              ← Win % vs random hand
• Pot odds: 33%                   ← Need 33% to call profitably
• +35% equity edge                ← You're ahead by this much

Rollout EVs:                      ← Simulated outcomes
  1/2 Pot Raise: +0.5 BB          ← Best action (+0.5 BB average)
  Call: +0.3 BB                   ← Okay but less profitable
  Fold: 0 BB                      ← Guaranteed loss of investment

• Low SPR - consider commitment   ← Strategic notes
• You have position               ← Positional advantage
```

## Key Concepts

### Equity
**Your chance of winning the hand**
- 50% = Coinflip
- 68% = Strong favorite
- 20% = Underdog (need good pot odds)

### Pot Odds
**Price to call vs pot size**
- Formula: `To Call / (Pot + To Call)`
- Example: Call 6 into pot of 12 = 6/18 = 33%
- **Rule**: Call if Equity > Pot Odds

### Expected Value (EV)
**Average profit/loss from action**
- +0.5 BB = Win half a big blind on average
- -0.2 BB = Lose on average
- 0 BB = Break even
- **Rule**: Choose highest EV action

### Stack-to-Pot Ratio (SPR)
**Stack size vs pot**
- SPR < 3 = Committed (hard to fold)
- SPR 3-10 = Medium (room to maneuver)
- SPR > 10 = Deep (many options)

### Position
**Who acts last**
- **In Position** = Act after opponent (advantage)
- **Out of Position** = Act first (disadvantage)
- **HU**: Button is in position postflop

## Training Progress

```
Episode 1000:   Exploring randomly (learning)
Episode 10000:  Learning basic concepts
Episode 30000:  Solid play emerging
Episode 50000:  Beating weak opponents
Episode 100000: Strong exploitative play
```

**Good signs**:
- Avg reward increasing
- Loss decreasing
- Epsilon decaying (less random)

**Bad signs**:
- Avg reward flat or negative
- Loss increasing or unstable
- → Reduce learning rate or change opponents

## Performance Benchmarks

| Opponent | Expected bb/100 | What It Means |
|----------|----------------|---------------|
| Random | +20 to +30 | Should easily beat |
| TAG | +5 to +15 | Solid performance |
| LAG | +10 to +20 | Punishing aggression |
| **Overall** | **+10 to +20** | **Profitable bot** |

**Comparison**:
- Break even: 0 bb/100
- Winning recreational: 3-5 bb/100
- Winning reg: 5-10 bb/100
- Crushing: 10+ bb/100

## File Locations

| Type | Location |
|------|----------|
| **Trained models** | `checkpoints/dqn_final.pt` |
| **Training script** | `experiments/holdem/train_holdem_dqn_vs_pool.py` |
| **GUI** | `gui/holdem_poker_gui.py` |
| **Environment** | `src/poker/envs/holdem_hu_env.py` |
| **Agent** | `src/poker/agents/holdem_dqn.py` |
| **Coach** | `src/poker/coach/agentic_coach.py` |
| **Docs** | `docs/holdem/*.md` |

## Common Issues

| Problem | Solution |
|---------|----------|
| "No module named gymnasium" | `pip install gymnasium` |
| "No trained agent found" | Train first or play vs random |
| Coach is slow | Reduce `n_rollouts` to 50 |
| Training is slow | Use GPU or reduce episodes |
| Bot plays randomly | Not trained enough yet |

## GUI Shortcuts

- **Space**: New hand
- **F**: Fold (when your turn)
- **C**: Check/Call (when your turn)
- **Enter**: Get coach advice

## Typical Workflow

```
Day 1: Install → Test → Play vs random
       ↓
Day 2: Train overnight (50k episodes)
       ↓
Day 3: Evaluate → Play vs trained bot → Study
       ↓
Day 4: Customize → Retrain → Improve
```

## Pro Tips

1. **Learning**:
   - Play 10 hands without coach
   - Then check what coach would recommend
   - Learn from differences

2. **Training**:
   - Start with 10k episodes for testing
   - Scale to 50k+ for serious bot
   - Save checkpoints every 5k

3. **Evaluation**:
   - Run 10k hands minimum
   - Check bb/100 across all opponents
   - Aim for consistent +EV

4. **Customization**:
   - Tweak hyperparameters
   - Add opponent types
   - Experiment with bet sizes

## Math Quick Reference

```
Equity needed for Call = Pot Odds
Pot Odds = To Call / (Pot + To Call)

Example:
  Pot = 12, To Call = 6
  Odds = 6 / 18 = 0.33 = 33%
  Need 33% equity to break even

If Equity = 68%:
  Profit = (68% - 33%) × 18 = 6.3 BB expected
```

## Next Steps

- [ ] Read SETUP_GUIDE.md for detailed instructions
- [ ] Read docs/holdem/holdem_design_notes.md for architecture
- [ ] Read docs/holdem/holdem_agentic_coach.md for coach details
- [ ] Check PROJECT_SUMMARY.md for overview

---

**Print this card and keep it nearby while using the system!** 📋
