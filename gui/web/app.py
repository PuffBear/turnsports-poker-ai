from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.poker.envs.holdem_hu_env import HoldemHuEnv
from src.poker.agents.holdem_dqn import DQNAgent
from src.poker.agents.holdem_policy_wrapper import PolicyWrapper
from src.poker.coach.agentic_coach import AgenticCoach
from src.poker.opponents.holdem_rule_based import RandomOpponent
import torch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'poker-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Game state
game_state = {
    'env': None,
    'coach': None,
    'bot_policy': None,
    'current_state': None
}

def init_game():
    """Initialize game environment and load bot."""
    game_state['env'] = HoldemHuEnv(stack_size=100.0, blinds=(0.5, 1.0))
    
    # Try to load trained bot
    bot_path = 'checkpoints/improved_dqn_final.pt'
    if not os.path.exists(bot_path):
        bot_path = 'checkpoints/dqn_final.pt'
    
    if os.path.exists(bot_path):
        agent = DQNAgent(state_dim=200, action_dim=9, device='cpu')
        agent.load(bot_path)
        game_state['bot_policy'] = PolicyWrapper(agent)
        print(f"✓ Loaded bot from {bot_path}")
    else:
        print("! No trained bot found, using random")
        game_state['bot_policy'] = None
    
    # Initialize coach
    game_state['coach'] = AgenticCoach(
        bot_policy=game_state['bot_policy'],
        use_rollouts=False,  # Disable for speed in web UI
        use_equity=True
    )

@app.route('/')
def index():
    return render_template('poker.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    init_game()
    emit('connected', {'status': 'ready'})

@socketio.on('new_hand')
def handle_new_hand():
    """Start a new hand."""
    env = game_state['env']
    state, info = env.reset()
    game_state['current_state'] = state
    
    # Get game state
    game_data = get_game_state()
    
    # If bot starts, let it act
    if env.current_player == 1:
        bot_action = get_bot_action()
        state, reward, done, truncated, info = env.step(bot_action)
        game_state['current_state'] = state
        game_data = get_game_state()
        game_data['bot_action'] = AgenticCoach.ACTION_NAMES[bot_action]
    
    emit('game_state', game_data)

@socketio.on('player_action')
def handle_player_action(data):
    """Handle player taking an action."""
    action_id = data['action_id']
    env = game_state['env']
    
    # Validate
    legal_actions = env._get_legal_actions()
    if action_id not in legal_actions:
        emit('error', {'message': 'Illegal action'})
        return
    
    # Execute player action
    state, reward, done, truncated, info = env.step(action_id)
    game_state['current_state'] = state
    
    # Get updated state
    game_data = get_game_state()
    
    if done:
        game_data['hand_complete'] = True
        game_data['result'] = {
            'reward': reward,
            'winner': info.get('winner', -1)
        }
        emit('game_state', game_data)
        return
    
    # Bot's turn
    bot_action = get_bot_action()
    state, reward, done, truncated, info = env.step(bot_action)
    game_state['current_state'] = state
    
    game_data = get_game_state()
    game_data['bot_action'] = AgenticCoach.ACTION_NAMES[bot_action]
    
    if done:
        game_data['hand_complete'] = True
        game_data['result'] = {
            'reward': reward,
            'winner': info.get('winner', -1)
        }
    
    emit('game_state', game_data)

@socketio.on('get_coach_advice')
def handle_coach_advice():
    """Get coach recommendation."""
    env = game_state['env']
    coach = game_state['coach']
    
    if env.current_player != 0:
        emit('coach_advice', {'error': 'Not your turn'})
        return
    
    try:
        recommendation = coach.get_recommendation(env, n_rollouts=0, n_equity_samples=2000)
        
        emit('coach_advice', {
            'action': recommendation['action_name'],
            'explanation': recommendation['explanation'],
            'tools': recommendation['tool_results']
        })
    except Exception as e:
        emit('coach_advice', {'error': str(e)})

def get_bot_action():
    """Get bot's action."""
    env = game_state['env']
    legal_actions = env._get_legal_actions()
    
    if game_state['bot_policy']:
        return game_state['bot_policy'].get_action(
            game_state['current_state'],
            legal_actions,
            deterministic=True
        )
    else:
        # Random opponent
        opponent = RandomOpponent()
        return opponent.get_action(env, game_state['current_state'])

def get_game_state():
    """Get current game state for frontend."""
    env = game_state['env']
    
    # Convert cards to strings
    player_hand = [str(c) for c in env.hands[0]]
    board = [str(c) for c in env.board]
    
    # Bot hand (hidden unless showdown)
    bot_hand_hidden = ['??', '??']
    
    return {
        'player_hand': player_hand,
        'bot_hand': bot_hand_hidden,
        'board': board,
        'pot': float(env.pot),
        'player_stack': float(env.stacks[0]),
        'bot_stack': float(env.stacks[1]),
        'street': ['Preflop', 'Flop', 'Turn', 'River'][env.street],
        'current_player': env.current_player,
        'legal_actions': env._get_legal_actions(),
        'to_call': float(abs(env.street_investment[0] - env.street_investment[1]))
    }

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🃏 POKER WEB GUI - Starting Server")
    print("=" * 60)
    print("\nOpen your browser to: http://localhost:5001")
    print("=" * 60 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
