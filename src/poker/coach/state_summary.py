def summarize_state(env):
    """
    Convert environment state to human-readable summary for the coach.
    Returns dict with key info.
    """
    player = env.current_player
    my_hand = env.hands[player]
    board = env.board
    pot = env.pot
    my_stack = env.stacks[player]
    opp_stack = env.stacks[1 - player]
    to_call = abs(env.street_bets[0] - env.street_bets[1])
    
    street_names = ['Preflop', 'Flop', 'Turn', 'River']
    street_name = street_names[env.street]
    
    # Position
    if player == env.button:
        position = "Button/SB"
        in_position = env.street > 0  # Postflop button is in position
    else:
        position = "BB"
        in_position = env.street == 0  # Preflop BB is in position
    
    # Format cards
    hand_str = ''.join([str(c) for c in my_hand])
    board_str = ''.join([str(c) for c in board]) if board else "None"
    
    # SPR (Stack to Pot Ratio)
    spr = my_stack / (pot + 0.01)  # Avoid division by zero
    
    summary = {
        'street': street_name,
        'hand': hand_str,
        'board': board_str,
        'pot': pot,
        'my_stack': my_stack,
        'opp_stack': opp_stack,
        'to_call': to_call,
        'position': position,
        'in_position': in_position,
        'spr': spr,
        'pot_odds': to_call / (pot + to_call) if to_call > 0 else 0
    }
    
    # Create text description
    text = f"Street: {street_name}\n"
    text += f"Your Hand: {hand_str}\n"
    text += f"Board: {board_str}\n"
    text += f"Pot: {pot:.1f}BB\n"
    text += f"Your Stack: {my_stack:.1f}BB\n"
    text += f"Opponent Stack: {opp_stack:.1f}BB\n"
    text += f"To Call: {to_call:.1f}BB\n"
    text += f"Position: {position}\n"
    text += f"SPR: {spr:.2f}\n"
    
    summary['text'] = text
    
    return summary
