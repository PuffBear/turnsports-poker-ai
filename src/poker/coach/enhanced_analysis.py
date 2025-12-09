"""
Enhanced poker analysis tools for the coach.
"""

def analyze_board_texture(board):
    """
    Analyze board texture for strategic insights.
    
    Returns dict with:
        - texture: 'dry', 'wet', or 'coordinated'
        - description: Human-readable description
        - threats: List of possible draws
    """
    if not board or len(board) < 3:
        return {
            'texture': 'none',
            'description': 'Preflop',
            'threats': []
        }
    
    suits = [card.suit for card in board]
    ranks = [card.rank for card in board]
    
    # Check for flush draw
    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    max_suit_count = max(suit_counts.values())
    
    # Check for straight possibilities
    rank_values = []
    for rank in ranks:
        if rank == 'A':
            rank_values.append(14)
        elif rank == 'K':
            rank_values.append(13)
        elif rank == 'Q':
            rank_values.append(12)
        elif rank == 'J':
            rank_values.append(11)
        elif rank == 'T':
            rank_values.append(10)
        else:
            rank_values.append(int(rank))
    
    rank_values_sorted = sorted(rank_values)
    max_gap = max(rank_values_sorted[-1] - rank_values_sorted[0], 5)
    
    # Check for pairs
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    has_pair = any(count >= 2 for count in rank_counts.values())
    
    # Determine texture
    threats = []
    
    if max_suit_count >= 3:
        threats.append('flush draw')
    
    if max_gap <= 4:
        threats.append('straight draw')
    
    if has_pair:
        threats.append('trips/full house possible')
    
    # Classify
    if len(threats) >= 2:
        texture = 'wet'
        description = f"Coordinated board with {', '.join(threats)}"
    elif len(threats) == 1:
        texture = 'semi-wet'
        description = f"Semi-coordinated with {threats[0]}"
    else:
        texture = 'dry'
        description = "Rainbow, disconnected board"
    
    return {
        'texture': texture,
        'description': description,
        'threats': threats
    }


def estimate_hand_range(opponent_actions, board, position):
    """
    Estimate opponent's likely hand range based on actions.
    
    Returns:
        dict: Hand categories with percentages
    """
    # Simplified range estimation
    range_estimate = {
        'premium_pairs': 0.0,  # AA-QQ
        'medium_pairs': 0.0,   # JJ-77
        'small_pairs': 0.0,    # 66-22
        'broadway': 0.0,        # AK, AQ, KQ
        'suited_connectors': 0.0,
        'weak_holdings': 0.0
    }
    
    # Very simplified logic based on position and aggression
    if position == 'Button':
        # Wider range from button
        range_estimate['premium_pairs'] = 0.15
        range_estimate['medium_pairs'] = 0.20
        range_estimate['broadway'] = 0.25
        range_estimate['suited_connectors'] = 0.20
        range_estimate['weak_holdings'] = 0.20
    else:
        # Tighter range from out of position
        range_estimate['premium_pairs'] = 0.25
        range_estimate['medium_pairs'] = 0.25
        range_estimate['broadway'] = 0.30
        range_estimate['suited_connectors'] = 0.10
        range_estimate['weak_holdings'] = 0.10
    
    return range_estimate


def format_opponent_tendencies(opponent_stats):
    """Format opponent stats into readable insights."""
    if opponent_stats['hands_played'] < 5:
        return "Not enough hands to profile opponent yet."
    
    insights = []
    
    # VPIP (Voluntarily Put $ In Pot)
    vpip_actions = opponent_stats['preflop_raises'] + opponent_stats['preflop_calls']
    vpip = (vpip_actions / opponent_stats['hands_played']) * 100 if opponent_stats['hands_played'] > 0 else 0
    
    if vpip < 20:
        insights.append(f"Very tight player (VPIP: {vpip:.0f}%) - give credit to their bets")
    elif vpip > 40:
        insights.append(f"Loose player (VPIP: {vpip:.0f}%) - can bluff more often")
    else:
        insights.append(f"Balanced player (VPIP: {vpip:.0f}%)")
    
    # Aggression
    pfr = (opponent_stats['preflop_raises'] / opponent_stats['hands_played']) * 100 if opponent_stats['hands_played'] > 0 else 0
    if pfr > vpip * 0.7:
        insights.append("Aggressive (raises frequently)")
    elif pfr < vpip * 0.3:
        insights.append("Passive (calls more than raises)")
    
    # C-bet fold tendency
    if opponent_stats['postflop_bets'] > 0:
        fold_to_bet_pct = (opponent_stats['postflop_folds_to_bet'] / opponent_stats['postflop_bets']) * 100
        if fold_to_bet_pct > 60:
            insights.append(f"Folds often to bets ({fold_to_bet_pct:.0f}%) - good bluffing target")
        elif fold_to_bet_pct < 30:
            insights.append(f"Rarely folds to bets ({fold_to_bet_pct:.0f}%) - value bet more, bluff less")
    
    return " • ".join(insights)
