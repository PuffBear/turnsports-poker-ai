import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.poker.core.cards import Card
from src.poker.core.deck import Deck
from src.poker.core.hand_eval import HandEvaluator

class HoldemHuEnv(gym.Env):
    """
    Heads-Up No-Limit Texas Hold'em Environment - FIXED VERSION
    Proper game logic with correct reward attribution and state transitions.
    """
    # Action encoding
    ACT_FOLD = 0
    ACT_CHECK_CALL = 1
    ACT_MIN_RAISE = 2
    ACT_RAISE_QUARTER_POT = 3
    ACT_RAISE_THIRD_POT = 4
    ACT_RAISE_HALF_POT = 5
    ACT_RAISE_THREE_QUARTER_POT = 6
    ACT_RAISE_POT = 7
    ACT_ALL_IN = 8
    
    STREET_PREFLOP = 0
    STREET_FLOP = 1
    STREET_TURN = 2
    STREET_RIVER = 3
    
    def __init__(self, stack_size=100.0, blinds=(0.5, 1.0)):
        super().__init__()
        self.initial_stack = stack_size
        self.sb_amount = blinds[0]
        self.bb_amount = blinds[1]
        
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=1, shape=(200,), dtype=np.float32)
        
        self.deck = Deck()
        self.reset()

    @property
    def street_bets(self):
        """Alias used by older coach/CLI code."""
        return self.street_investment

    @street_bets.setter
    def street_bets(self, value):
        self.street_investment = value

    def get_to_call(self):
        """Return chips required to call the current bet."""
        return abs(self.street_investment[0] - self.street_investment[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize
        self.deck.reset()
        self.stacks = [self.initial_stack, self.initial_stack]
        self.hands = [self.deck.draw(2), self.deck.draw(2)]
        self.board = []
        self.pot = 0
        self.street = self.STREET_PREFLOP
        self.done = False
        self.rewards = None
        
        # Randomize button (dealer)
        self.button = np.random.randint(0, 2)
        self.sb_idx = self.button
        self.bb_idx = 1 - self.button
        
        # Post blinds
        sb_post = min(self.sb_amount, self.stacks[self.sb_idx])
        bb_post = min(self.bb_amount, self.stacks[self.bb_idx])
        
        self.stacks[self.sb_idx] -= sb_post
        self.stacks[self.bb_idx] -= bb_post
        self.pot = sb_post + bb_post
        
        # Betting state
        self.street_investment = [sb_post, bb_post]  # Total put in this street
        self.has_acted = [False, False]  # Track who has acted
        
        # Preflop: SB acts first
        self.current_player = self.sb_idx
        
        return self._get_obs(), {}

    def step(self, action_id):
        if self.done:
            return self._get_obs(), 0, True, False, {"done": True}
        
        # Validate action
        legal_actions = self._get_legal_actions()
        if action_id not in legal_actions:
            # Penalize illegal action
            self.done = True
            info = {
                "illegal_action": True,
                "winner": 1 - self.current_player,
                "pot": self.pot,
                "stacks": list(self.stacks),
                "board": list(self.board),
            }
            return self._get_obs(), -self.initial_stack, True, False, info
        
        # Execute action
        reward, done, info = self._execute_action(action_id)
        
        if not done:
            # Switch player
            self.current_player = 1 - self.current_player
        
        return self._get_obs(), reward, done, False, info

    def _execute_action(self, action_id):
        """Execute an action and return (reward, done, info)."""
        current_invest = self.street_investment[self.current_player]
        opp_invest = self.street_investment[1 - self.current_player]
        to_call = opp_invest - current_invest
        stack = self.stacks[self.current_player]
        
        # Mark that this player has acted
        self.has_acted[self.current_player] = True
        
        if action_id == self.ACT_FOLD:
            # Opponent wins
            return self._end_hand_fold()
        
        elif action_id == self.ACT_CHECK_CALL:
            # Call the difference
            call_amount = min(to_call, stack)
            self.stacks[self.current_player] -= call_amount
            self.street_investment[self.current_player] += call_amount
            self.pot += call_amount
            
            # Check if betting round is complete
            if self._is_betting_complete():
                return self._advance_street()
            
            return 0, False, {}
        
        else:  # Raise
            # Calculate raise amount
            pot_for_calc = self.pot + to_call
            
            if action_id == self.ACT_MIN_RAISE:
                raise_to = opp_invest + self.bb_amount
            elif action_id == self.ACT_RAISE_QUARTER_POT:
                raise_to = opp_invest + int(0.25 * pot_for_calc)
            elif action_id == self.ACT_RAISE_THIRD_POT:
                raise_to = opp_invest + int(0.33 * pot_for_calc)
            elif action_id == self.ACT_RAISE_HALF_POT:
                raise_to = opp_invest + int(0.5 * pot_for_calc)
            elif action_id == self.ACT_RAISE_THREE_QUARTER_POT:
                raise_to = opp_invest + int(0.75 * pot_for_calc)
            elif action_id == self.ACT_RAISE_POT:
                raise_to = opp_invest + pot_for_calc
            else:  # ALL_IN
                raise_to = current_invest + stack
            
            # Amount to put in
            amount = min(raise_to - current_invest, stack)
            
            # Must be at least a min-raise unless all-in
            if amount < stack and amount < to_call + self.bb_amount:
                amount = min(to_call + self.bb_amount, stack)
            
            self.stacks[self.current_player] -= amount
            self.street_investment[self.current_player] += amount
            self.pot += amount
            
            # Reset opponent's action flag (they need to respond to raise)
            self.has_acted[1 - self.current_player] = False
            
            # Check if someone is all-in
            if self.stacks[self.current_player] == 0 or self.stacks[1 - self.current_player] == 0:
                # Run out remaining streets
                return self._runout_and_showdown()
            
            return 0, False, {}

    def _is_betting_complete(self):
        """Check if betting round is complete."""
        # Both players must have acted
        if not all(self.has_acted):
            return False
        
        # Investments must be equal (or one is all-in)
        inv0, inv1 = self.street_investment
        if abs(inv0 - inv1) < 0.01:  # Equal (within floating point tolerance)
            return True
        
        # One player is all-in
        if self.stacks[0] == 0 or self.stacks[1] == 0:
            return True
        
        return False

    def _advance_street(self):
        """Advance to next street."""
        if self.street == self.STREET_RIVER:
            return self._showdown()
        
        # Deal next street
        self.street += 1
        if self.street == self.STREET_FLOP:
            self.board.extend(self.deck.draw(3))
        else:
            self.board.extend(self.deck.draw(1))
        
        # Reset betting for new street
        self.street_investment = [0, 0]
        self.has_acted = [False, False]
        
        # Postflop: BB acts first (out of position)
        self.current_player = self.bb_idx
        
        return 0, False, {}

    def _runout_and_showdown(self):
        """Deal remaining cards and go to showdown."""
        while len(self.board) < 5:
            if self.street == self.STREET_PREFLOP:
                self.board.extend(self.deck.draw(3))
                self.street = self.STREET_FLOP
            else:
                self.board.extend(self.deck.draw(1))
                self.street += 1
        
        return self._showdown()

    def _showdown(self):
        """Evaluate hands and return reward + info."""
        score0 = HandEvaluator.evaluate(self.hands[0] + self.board)
        score1 = HandEvaluator.evaluate(self.hands[1] + self.board)
        
        if score0 > score1:
            winner = 0
        elif score1 > score0:
            winner = 1
        else:
            winner = -1  # Tie
        
        # Distribute pot
        if winner == -1:
            # Split pot
            self.stacks[0] += self.pot / 2
            self.stacks[1] += self.pot / 2
        else:
            self.stacks[winner] += self.pot
        
        # Calculate final rewards (profit/loss for this hand)
        reward0 = self.stacks[0] - self.initial_stack
        reward1 = self.stacks[1] - self.initial_stack
        
        self.done = True
        self.rewards = [reward0, reward1]

        info = {
            "winner": winner,
            "hands": [list(self.hands[0]), list(self.hands[1])],
            "board": list(self.board),
            "pot": self.pot,
            "stacks": list(self.stacks),
            "hand_scores": [score0, score1],
            "street": self.street,
            "to_call": self.get_to_call(),
            "done": True,
        }
        
        # Return reward for CURRENT player (the one who just acted)
        return (reward0 if self.current_player == 0 else reward1), True, info

    def _end_hand_fold(self):
        """Handle fold - opponent wins."""
        winner = 1 - self.current_player
        self.stacks[winner] += self.pot
        
        reward0 = self.stacks[0] - self.initial_stack
        reward1 = self.stacks[1] - self.initial_stack
        
        self.done = True
        self.rewards = [reward0, reward1]

        info = {
            "winner": winner,
            "hands": [list(self.hands[0]), list(self.hands[1])],
            "board": list(self.board),
            "pot": self.pot,
            "stacks": list(self.stacks),
            "folded": self.current_player,
            "street": self.street,
            "to_call": self.get_to_call(),
            "done": True,
        }
        
        return (reward0 if self.current_player == 0 else reward1), True, info

    def _get_legal_actions(self):
        """Return list of legal action IDs."""
        actions = []
        
        current_invest = self.street_investment[self.current_player]
        opp_invest = self.street_investment[1 - self.current_player]
        to_call = opp_invest - current_invest
        stack = self.stacks[self.current_player]
        
        #  Can fold if facing a bet
        if to_call > 0:
            actions.append(self.ACT_FOLD)
        
        # Can always check/call
        actions.append(self.ACT_CHECK_CALL)
        
        # Can raise if have chips left after calling
        if stack > to_call:
            actions.extend([
                self.ACT_MIN_RAISE,
                self.ACT_RAISE_QUARTER_POT,
                self.ACT_RAISE_THIRD_POT,
                self.ACT_RAISE_HALF_POT,
                self.ACT_RAISE_THREE_QUARTER_POT,
                self.ACT_RAISE_POT,
                self.ACT_ALL_IN
            ])
        
        return actions

    def _get_obs(self):
        """Get observation vector with proper card and game state encoding."""
        obs = np.zeros(200, dtype=np.float32)
        idx = 0
        
        # 1. Hole cards (2 cards × 17 features = 34)
        # Each card: 13 rank one-hot + 4 suit one-hot
        for card in self.hands[self.current_player]:
            # Rank one-hot (13)
            obs[idx + card.rank_index] = 1
            idx += 13
            # Suit one-hot (4)
            obs[idx + card.suit_index] = 1
            idx += 4
        
        # 2. Board cards (5 cards × 17 features = 85)
        for i in range(5):
            if i < len(self.board):
                card = self.board[i]
                # Rank one-hot
                obs[idx + card.rank_index] = 1
                idx += 13
                # Suit one-hot
                obs[idx + card.suit_index] = 1
                idx += 4
            else:
                # Padding for unrevealed cards
                idx += 17
        
        # idx should now be at 119 (34 + 85)
        
        # 3. Pot and stack info (10)
        obs[idx] = min(self.pot / 200, 1.0)  # Normalized pot
        obs[idx + 1] = self.stacks[self.current_player] / 100  # My stack
        obs[idx + 2] = self.stacks[1 - self.current_player] / 100  # Opp stack
        
        # Amount to call
        to_call = abs(self.street_investment[0] - self.street_investment[1])
        obs[idx + 3] = min(to_call / 100, 1.0)
        
        # SPR (stack to pot ratio)
        spr = self.stacks[self.current_player] / (self.pot + 1e-6)
        obs[idx + 4] = min(spr / 20, 1.0)  # Cap at 20
        
        # My investment this street
        obs[idx + 5] = min(self.street_investment[self.current_player] / 100, 1.0)
        
        # Opponent investment this street
        obs[idx + 6] = min(self.street_investment[1 - self.current_player] / 100, 1.0)
        
        # Pot odds if facing bet
        if to_call > 0:
            pot_odds = to_call / (self.pot + to_call)
            obs[idx + 7] = pot_odds
        
        idx += 10  # Now at 129
        
        # 4. Street (4 one-hot)
        obs[idx + self.street] = 1
        idx += 4  # Now at 133
        
        # 5. Position (2)
        obs[idx] = 1 if self.current_player == self.button else 0  # Am I button?
        obs[idx + 1] = 1 if self.current_player == self.bb_idx else 0  # Am I BB?
        idx += 2  # Now at 135
        
        # Rest zeros (65 dimensions reserved for future use: betting history etc.)
        
        return obs

    def render(self):
        print(f"Street: {['PREFLOP', 'FLOP', 'TURN', 'RIVER'][self.street]}")
        print(f"Board: {self.board}")
        print(f"Pot: {self.pot:.1f} BB")
        print(f"P0: {self.hands[0]} Stack: {self.stacks[0]:.1f} BB")
        print(f"P1: {self.hands[1]} Stack: {self.stacks[1]:.1f} BB")
