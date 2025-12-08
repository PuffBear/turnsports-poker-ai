# Ollama Setup for LLM Coach

The CFR Poker GUI includes an LLM-powered recommendation system that uses Ollama to run local Llama models.

## Installation Steps

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [https://ollama.com/download/windows](https://ollama.com/download/windows)

### 2. Start Ollama Server

The Ollama server should start automatically after installation. If not, run:
```bash
ollama serve
```

### 3. Download a Model

Download a Llama model (recommended: llama3.2 for speed, or llama2 for compatibility):

```bash
ollama pull llama3.2
```

Or for Llama 2:
```bash
ollama pull llama2
```

Other models you can try:
- `mistral` - Fast and efficient
- `llama3` - Larger, more capable
- `phi3` - Microsoft's efficient model

### 4. Install Python Package

The `ollama` Python package is already in `requirements.txt`. Install it:

```bash
pip install ollama
```

Or if using the venv:
```bash
./.venv-tk12/bin/pip install ollama
```

### 5. Run the GUI

```bash
./.venv-tk12/bin/python3.12 gui/cfr_poker_gui.py
```

The LLM coach will automatically detect if Ollama is running and use it for recommendations. If Ollama is not available, it will fall back to template-based recommendations.

## Changing the Model

You can change the model used by editing `gui/cfr_poker_gui.py` and modifying the model parameter:

```python
self.llm_coach = LLMCoach(bot_policy=bot_policy, risk_tolerance='moderate', model='llama3.2')
```

## Troubleshooting

**"Ollama server not running"**
- Make sure Ollama is installed: `ollama --version`
- Start the server: `ollama serve`
- Check if it's running: `curl http://localhost:11434/api/tags`

**"Model not found"**
- Pull the model: `ollama pull llama3.2`
- List available models: `ollama list`

**Slow responses**
- Use a smaller model like `llama3.2` or `phi3`
- Reduce `num_predict` in the code (currently 200 tokens)

**Memory issues**
- Use a smaller model
- Close other applications
- Consider using `phi3` or `mistral` which are more memory-efficient

## Features

The LLM coach provides:
- **Natural language recommendations** based on game state
- **Risk tolerance levels**: Conservative, Moderate, Aggressive
- **Statistics**: Equity, pot odds, SPR, rollout EVs
- **Strategic analysis** for all risk levels

Enjoy your AI-powered poker coaching!

