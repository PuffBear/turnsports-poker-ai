# Beating the Boss: Deep RL Poker Bot and Agentic AI Coach

🃏 **Heads-Up No-Limit Texas Hold'em** with Deep Q-Learning and an AI Coach

## Project Overview

This project implements:

1. **RL Poker Bot**: A Deep Q-Network (DQN) agent trained to play Heads-Up No-Limit Texas Hold'em
2. **Agentic AI Coach**: An intelligent coach that uses tools (equity estimation, rollout simulation) to provide strategic recommendations
3. **Interactive GUI**: Play against the bot with real-time coaching assistance

## Features

### Poker Environment
- **Game**: Heads-Up No-Limit Texas Hold'em
- **Stack Size**: 100BB effective
- **Streets**: Preflop → Flop → Turn → River
- **Bet Abstraction**: 
  - Fold
  - Check/Call
  - Min-raise
  - 1/4 pot, 1/3 pot, 1/2 pot, 3/4 pot, pot-sized raises
  - All-in

### RL Bot
- **Algorithm**: Deep Q-Network (DQN) with experience replay and target networks
- **Training**: Self-play and vs. rule-based opponents (Random, TAG, LAG)
- **State Representation**: Cards, pot, stacks, betting history, position
- **Action Space**: 9 discrete actions with legality filtering

### Agentic Coach
The coach uses multiple tools to analyze situations:
- **Equity Estimator**: Monte Carlo simulation to estimate win probability
- **Pot Odds Calculator**: Computes required equity for profitable calls
- **Rollout Simulator**: Simulates different action lines vs. the bot
- **Strategic Reasoning**: Considers SPR, position, and exploit opportunities

## Installation

```bash
# Clone repository
git clone <repo-url>
cd turnsports-poker-ai

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Bot

```bash
python experiments/holdem/train_holdem_dqn_vs_pool.py
```

This will train a DQN agent for 50,000 episodes against a pool of opponents. Checkpoints are saved every 5,000 episodes.

### 2. Play Against the Bot

```bash
python gui/holdem_poker_gui.py
```

The GUI provides:
- **Left Panel**: Game state, your cards, action buttons
- **Right Panel**: AI Coach recommendations and analysis

### 3. Get Coach Advice

Click "Get Coach Advice" to receive:
- Recommended action
- Equity analysis
- Rollout EV estimates for different actions
- Strategic considerations (SPR, position, etc.)

## Project Structure

```
poker-rl-cfr/
├── src/poker/
│   ├── core/           # Card, Deck, Hand evaluation
│   ├── envs/           # Kuhn Poker & Hold'em environments
│   ├── agents/         # DQN agent implementation
│   ├── opponents/      # Rule-based opponents
│   ├── coach/          # Agentic coach with tools
│   └── analysis/       # Evaluation utilities
├── experiments/        # Training scripts
├── gui/               # Interactive GUI
├── docs/              # Documentation
└── notebooks/         # Analysis notebooks
```

## Key Components

### Environment (`src/poker/envs/holdem_hu_env.py`)
Gymnasium-compatible environment for HU NLHE with:
- Proper blind structure
- Betting round logic
- Showdown evaluation
- State vectorization for RL

### DQN Agent (`src/poker/agents/holdem_dqn.py`)
- 3-layer fully connected network
- Experience replay buffer (100k transitions)
- Target network for stable training
- Epsilon-greedy exploration

### AI Coach (`src/poker/coach/agentic_coach.py`)
Modular coach that:
1. Summarizes game state
2. Calls equity tools
3. Runs rollout simulations
4. Generates natural language explanations

## Training Details

**Phase A - vs Rule-Based Opponents**
- Train vs. mix of Random, TAG, LAG opponents
- Learn basic concepts: value betting, folding trash hands

**Phase B - Self-Play** (Future work)
- Maintain pool of agent snapshots
- Fictitious self-play for game-theoretic learning

**Hyperparameters**:
- Learning rate: 1e-4
- Gamma: 0.99
- Epsilon: 1.0 → 0.1 (decay 0.9995)
- Batch size: 64
- Replay buffer: 100k

## Usage Examples

### Training with Custom Settings

```python
from experiments.holdem.train_holdem_dqn_vs_pool import train_dqn_vs_pool

agent = train_dqn_vs_pool(
    n_episodes=100000,
    save_interval=10000,
    device='cuda'
)
```

### Using the Coach Programmatically

```python
from src.poker.coach.agentic_coach import AgenticCoach
from src.poker.agents.holdem_policy_wrapper import PolicyWrapper

# Load trained agent
agent = DQNAgent(state_dim=200, action_dim=9)
agent.load('checkpoints/dqn_final.pt')
bot_policy = PolicyWrapper(agent)

# Initialize coach
coach = AgenticCoach(bot_policy=bot_policy)

# Get recommendation
recommendation = coach.get_recommendation(env)
print(recommendation['explanation'])
```

## Evaluation

Evaluate the bot's performance:

```bash
python experiments/holdem/eval_bot_vs_baselines.py
```

Metrics:
- **bb/100**: Big blinds won per 100 hands
- **Win rate**: Percentage of hands won
- **Showdown equity**: Performance at showdown

## Future Work

- [ ] Implement full self-play training
- [ ] Add LLM integration for natural language coaching
- [ ] Expand bet abstraction
- [ ] Monte Carlo CFR for game-theoretic optimality
- [ ] Hand range visualization
- [ ] Tournament mode

## Connection to Kuhn Poker

This project extends concepts from Kuhn Poker (toy game) to full Texas Hold'em:
- **Kuhn**: 3 cards, 2 players, simple actions → solvable with CFR
- **Hold'em**: 52 cards, 4 betting rounds, continuous bets → requires function approximation (DQN)

See `docs/kuhn/` for the foundational work on simplified poker games.

## Contributing

Contributions welcome! Areas of interest:
- Advanced RL algorithms (A3C, PPO, CFR+)
- Better hand evaluation
- UI/UX improvements
- More sophisticated opponents

## License

MIT

## Acknowledgments

- Environment design inspired by OpenAI Gym
- DQN implementation based on Mnih et al. (2015)
- Hand evaluation using standard poker rules
- Coach architecture inspired by modern agentic AI systems

---

**Play smart. Learn from the coach. Beat the boss!** 🃏🤖