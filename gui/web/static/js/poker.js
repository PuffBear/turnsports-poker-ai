const socket = io();

// Game state
let gameState = null;

// Connect to server
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('connected', (data) => {
    console.log('Game ready:', data);
    document.getElementById('new-hand-btn').disabled = false;
});

// Game state updates
socket.on('game_state', (data) => {
    console.log('Game state:', data);
    gameState = data;
    updateUI(data);
});

// Coach advice
socket.on('coach_advice', (data) => {
    console.log('Coach advice:', data);
    updateCoachPanel(data);
});

// Error handling
socket.on('error', (data) => {
    console.error('Error:', data);
    alert('Error: ' + data.message);
});

// UI Updates
function updateUI(state) {
    // Update player hand
    const playerHand = document.getElementById('player-hand');
    playerHand.innerHTML = state.player_hand.map(card => 
        `<div class="card">${formatCard(card)}</div>`
    ).join('');
    
    // Update bot hand (hidden)
    const botHand = document.getElementById('bot-hand');
    botHand.innerHTML = state.bot_hand.map(card => 
        `<div class="card card-back">${card}</div>`
    ).join('');
    
    // Update board
    const board = document.getElementById('board');
    board.innerHTML = '';
    for (let i = 0; i < 5; i++) {
        if (i < state.board.length) {
            board.innerHTML += `<div class="card">${formatCard(state.board[i])}</div>`;
        } else {
            board.innerHTML += `<div class="card-slot"></div>`;
        }
    }
    
    // Update pot
    document.getElementById('pot-amount').textContent = `${state.pot.toFixed(1)} BB`;
    
    // Update stacks
    document.getElementById('player-stack').textContent = `${state.player_stack.toFixed(1)} BB`;
    document.getElementById('bot-stack').textContent = `${state.bot_stack.toFixed(1)} BB`;
    
    // Update street
    document.getElementById('street-label').textContent = state.street;
    
    // Update call button text
    const callBtn = document.querySelector('[data-action="1"]');
    if (state.to_call > 0) {
        callBtn.querySelector('#call-text').textContent = `Call ${state.to_call.toFixed(1)}`;
    } else {
        callBtn.querySelector('#call-text').textContent = 'Check';
    }
    
    // Enable/disable action buttons
    const actionButtons = document.querySelectorAll('.action-controls .btn');
    actionButtons.forEach(btn => {
        const actionId = parseInt(btn.getAttribute('data-action'));
        btn.disabled = !state.legal_actions.includes(actionId) || state.current_player !== 0;
    });
    
    // Show bot action if any
    if (state.bot_action) {
        showNotification(`Bot: ${state.bot_action}`);
    }
    
    // Handle hand completion
    if (state.hand_complete) {
        handleHandComplete(state.result);
    }
}

function formatCard(cardStr) {
    // Format card string (e.g., "As" -> "A♠")
    if (!cardStr || cardStr === '??') return cardStr;
    
    const rank = cardStr[0];
    const suit = cardStr[1];
    
    const suitSymbols = {
        's': '♠',
        'h': '♥',
        'd': '♦',
        'c': '♣'
    };
    
    const symbol = suitSymbols[suit] || suit;
    
    // Color the card based on suit
    const isRed = suit === 'h' || suit === 'd';
    const color = isRed ? 'style="color: #ef4444"' : 'style="color: #1f2937"';
    
    return `<span ${color}>${rank}${symbol}</span>`;
}

function updateCoachPanel(data) {
    if (data.error) {
        document.getElementById('recommended-action').textContent = data.error;
        return;
    }
    
    // Update recommendation
    document.getElementById('recommended-action').textContent = data.action;
    
    // Update explanation
    document.getElementById('coach-explanation').textContent = data.explanation;
    
    // Update stats
    if (data.tools) {
        if (data.tools.equity !== undefined) {
            document.getElementById('equity-stat').textContent = 
                `${(data.tools.equity * 100).toFixed(1)}%`;
        }
        if (data.tools.pot_odds !== undefined) {
            document.getElementById('pot-odds-stat').textContent = 
                `${(data.tools.pot_odds * 100).toFixed(1)}%`;
        }
    }
}

function handleHandComplete(result) {
    let message = '';
    if (result.winner === 0) {
        message = `🎉 You won ${result.reward.toFixed(1)} BB!`;
    } else if (result.winner === 1) {
        message = `😔 Bot won ${Math.abs(result.reward).toFixed(1)} BB`;
    } else {
        message = '🤝 Split pot';
    }
    
    setTimeout(() => {
        showNotification(message);
    }, 500);
}

function showNotification(message) {
    // Simple notification (you can make this fancier)
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(0,0,0,0.9);
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        font-size: 16px;
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Event Listeners
document.getElementById('new-hand-btn').addEventListener('click', () => {
    socket.emit('new_hand');
});

document.getElementById('get-advice-btn').addEventListener('click', () => {
    socket.emit('get_coach_advice');
});

// Action buttons
document.querySelectorAll('.action-controls .btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const actionId = parseInt(btn.getAttribute('data-action'));
        socket.emit('player_action', { action_id: actionId });
    });
});

// Initialize
document.getElementById('new-hand-btn').disabled = true;
