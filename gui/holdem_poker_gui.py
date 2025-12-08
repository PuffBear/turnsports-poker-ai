import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import tkinter as tk
except ModuleNotFoundError as e:
    if e.name == "_tkinter":
        raise SystemExit(
            "Tkinter is not available in this Python build.\n"
            "Install a Tk-enabled Python (e.g., python.org installer or brew tcl-tk + reinstall python)"
        ) from e
    else:
        raise

import random
from src.poker.core.deck import Deck
from src.poker.core.hand_eval import HandEvaluator

# Minimal self-contained heads-up hold'em environment (no gym/numpy)
class SimpleHoldemEnv:
    ACT_FOLD = 0
    ACT_CHECK_CALL = 1
    ACT_RAISE_HALF_POT = 5  # align with original ID
    ACT_MIN_RAISE = 5       # alias for callers

    STREET_PREFLOP = 0
    STREET_FLOP = 1
    STREET_TURN = 2
    STREET_RIVER = 3

    def __init__(self, stack_size=100.0, blinds=(0.5, 1.0)):
        self.initial_stack = stack_size
        self.sb_amount = blinds[0]
        self.bb_amount = blinds[1]
        self.deck = Deck()
        self.reset()

    def get_to_call(self):
        return abs(self.street_investment[0] - self.street_investment[1])

    def reset(self):
        self.deck.reset()
        self.stacks = [self.initial_stack, self.initial_stack]
        self.hands = [self.deck.draw(2), self.deck.draw(2)]
        self.board = []
        self.pot = 0
        self.street = self.STREET_PREFLOP
        self.done = False
        self.rewards = None

        # For GUI simplicity: player is always button/SB and acts first
        self.button = 0
        self.sb_idx = 0
        self.bb_idx = 1

        # Post blinds
        sb_post = min(self.sb_amount, self.stacks[self.sb_idx])
        bb_post = min(self.bb_amount, self.stacks[self.bb_idx])
        self.stacks[self.sb_idx] -= sb_post
        self.stacks[self.bb_idx] -= bb_post
        self.pot = sb_post + bb_post
        self.street_investment = [sb_post, bb_post]
        self.has_acted = [False, False]

        # Preflop: SB acts first
        self.current_player = self.sb_idx
        return self._get_obs(), {}

    def _get_obs(self):
        # For bot; we won't use numerical obs; just return dict
        return {
            "hands": self.hands,
            "board": self.board,
            "stacks": self.stacks,
            "pot": self.pot,
            "street": self.street,
            "current_player": self.current_player,
            "to_call": self.get_to_call(),
        }

    def _get_legal_actions(self):
        acts = [self.ACT_CHECK_CALL]
        if self.get_to_call() > 0:
            acts.append(self.ACT_FOLD)
        if self.stacks[self.current_player] > 0:
            acts.append(self.ACT_RAISE_HALF_POT)
        return acts

    def step(self, action_id):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        current = self.current_player
        opp = 1 - current
        to_call = self.get_to_call()
        info = {}
        reward = 0

        if action_id == self.ACT_FOLD:
            winner = opp
            self.stacks[winner] += self.pot
            self.rewards = [self.stacks[0] - self.initial_stack,
                            self.stacks[1] - self.initial_stack]
            self.done = True
            info["winner"] = winner
            return self._get_obs(), self.rewards[current], True, False, info

        if action_id == self.ACT_CHECK_CALL:
            call_amt = min(to_call, self.stacks[current])
            self.stacks[current] -= call_amt
            self.street_investment[current] += call_amt
            self.pot += call_amt
            self.has_acted[current] = True
            if self._betting_complete():
                rew, dn, info = self._advance_street()
                return self._get_obs(), rew, dn, False, info
            else:
                self.current_player = opp
                return self._get_obs(), 0, False, False, {}

        # Raise half-pot
        raise_to = self.street_investment[opp] + max(to_call + self.bb_amount,
                                                     0.5 * (self.pot + to_call))
        amt = min(raise_to - self.street_investment[current], self.stacks[current])
        if amt < to_call + self.bb_amount and amt < self.stacks[current]:
            amt = min(to_call + self.bb_amount, self.stacks[current])

        self.stacks[current] -= amt
        self.street_investment[current] += amt
        self.pot += amt
        self.has_acted[current] = True
        self.has_acted[opp] = False  # opponent must respond

        if self.stacks[current] == 0 or self.stacks[opp] == 0:
            rew, dn, info = self._runout_and_showdown()
            return self._get_obs(), rew, dn, False, info

        self.current_player = opp
        return self._get_obs(), 0, False, False, {}

    def _betting_complete(self):
        if not all(self.has_acted):
            return False
        if abs(self.street_investment[0] - self.street_investment[1]) < 1e-6:
            return True
        if self.stacks[0] == 0 or self.stacks[1] == 0:
            return True
        return False

    def _advance_street(self):
        if self.street == self.STREET_RIVER:
            return self._showdown()
        self.street += 1
        if self.street == self.STREET_FLOP:
            self.board.extend(self.deck.draw(3))
        else:
            self.board.extend(self.deck.draw(1))
        self.street_investment = [0, 0]
        self.has_acted = [False, False]
        # Postflop: BB acts first
        self.current_player = self.bb_idx
        return 0, False, {}

    def _runout_and_showdown(self):
        while len(self.board) < 5:
            self.board.extend(self.deck.draw(1 if len(self.board) >= 3 else 3))
        return self._showdown()

    def _showdown(self):
        score0 = HandEvaluator.evaluate(self.hands[0] + self.board)
        score1 = HandEvaluator.evaluate(self.hands[1] + self.board)
        if score0 > score1:
            winner = 0
        elif score1 > score0:
            winner = 1
        else:
            winner = -1

        if winner == -1:
            self.stacks[0] += self.pot / 2
            self.stacks[1] += self.pot / 2
        else:
            self.stacks[winner] += self.pot

        self.rewards = [self.stacks[0] - self.initial_stack,
                        self.stacks[1] - self.initial_stack]
        self.done = True
        info = {"winner": winner, "hands": self.hands, "board": self.board}
        return self.rewards[self.current_player], True, info


class SimpleRandomBot:
    """Bot that picks uniformly from legal actions."""
    def get_action(self, env, obs):
        legal = env._get_legal_actions()
        return random.choice(legal) if legal else env.ACT_CHECK_CALL

class HoldemPokerGUI:
    """
    Fully functional 1v1 Texas Hold'em GUI with real gameplay.
    """
    def __init__(self, root, agent_path=None):
        self.root = root
        self.root.title("Texas Hold'em - 1v1 Table")
        self.root.geometry("1200x780")
        self.root.configure(bg='#f5f5f5')
        
        # Initialize environment (minimal, no numpy/gym required)
        self.env = SimpleHoldemEnv(stack_size=100.0, blinds=(0.5, 1.0))
        
        # Initialize bot opponent (simple random)
        self.bot = SimpleRandomBot()
        self.player_idx = 0
        self.bot_idx = 1
        
        # Canvas for table rendering
        self.canvas = tk.Canvas(self.root, bg="#f5f5f5", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # Alias for compatibility
        self.update_display = self.redraw

        # Action buttons
        btn_frame = tk.Frame(self.root, bg="#f5f5f5")
        btn_frame.pack(pady=12)
        self.fold_btn = tk.Button(btn_frame, text="Fold", width=12, height=2,
                                  font=("Arial", 12), command=self.action_fold)
        self.fold_btn.pack(side=tk.LEFT, padx=8)
        self.call_btn = tk.Button(btn_frame, text="Check/Call", width=12, height=2,
                                  font=("Arial", 12), command=self.action_call)
        self.call_btn.pack(side=tk.LEFT, padx=8)
        self.raise_btn = tk.Button(btn_frame, text="Raise (1/2 Pot)", width=15, height=2,
                                   font=("Arial", 12), command=self.action_raise)
        self.raise_btn.pack(side=tk.LEFT, padx=8)
        self.new_hand_btn = tk.Button(btn_frame, text="New Hand", width=12, height=2,
                                      font=("Arial", 12), command=self.new_hand)
        self.new_hand_btn.pack(side=tk.LEFT, padx=8)

        # Status label
        self.status_label = tk.Label(self.root, text="", font=("Arial", 13, "bold"),
                                     bg="#f5f5f5", fg="#333333", wraplength=950)
        self.status_label.pack(pady=8)
    
    def card_to_str(self, card):
        """Convert card object to display string."""
        if card is None:
            return "??"
        return str(card)
    
    def redraw(self):
        """Redraw the whole table on the canvas."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 1200
        h = self.canvas.winfo_height() or 780
        cx = w // 2
        # Slightly tighter vertical stack so labels are not covered
        top_y = 105
        mid_y = h // 2 - 60
        bot_y = h - 210

        # Table outline
        margin = 70
        self.canvas.create_oval(margin, margin, w - margin, h - margin,
                                fill="#c6f0c6", outline="#7bbf7b", width=4)

        # Board
        self.canvas.create_text(cx, mid_y - 70, text="BOARD", fill="#111111", font=("Arial", 18, "bold"))
        self._draw_card_row(cx, mid_y - 10, self.env.board, up=True, total_slots=5, card_width=70)

        # Pot and street
        street_names = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
        street_txt = street_names[self.env.street] if self.env.street < len(street_names) else "UNKNOWN"
        self.canvas.create_text(cx, mid_y + 35, text=f"Pot: {self.env.pot:.1f} BB",
                                fill="#0044aa", font=("Arial", 16, "bold"))
        self.canvas.create_text(cx, mid_y + 58, text=f"Street: {street_txt}",
                                fill="#111111", font=("Arial", 14, "bold"))
        to_call = self.env.get_to_call()
        if to_call > 0:
            self.canvas.create_text(cx, mid_y + 81, text=f"To Call: {to_call:.1f} BB",
                                    fill="#cc6600", font=("Arial", 13, "bold"))

        # Bot seat (top)
        self.canvas.create_text(cx, top_y - 10, text="BOT", fill="#111111", font=("Arial", 18, "bold"))
        self._draw_card_row(cx, top_y + 30, ["??", "??"], up=False, card_width=70)
        self.canvas.create_text(cx, top_y + 65, text=f"Stack: {self.env.stacks[self.bot_idx]:.1f} BB",
                                fill="#111111", font=("Arial", 13, "bold"))

        # Player seat (bottom)
        self.canvas.create_text(cx, bot_y, text="YOU", fill="#111111", font=("Arial", 18, "bold"))
        self._draw_card_row(cx, bot_y + 30, self.env.hands[self.player_idx], up=True, card_width=70)
        self.canvas.create_text(cx, bot_y + 60, text=f"Stack: {self.env.stacks[self.player_idx]:.1f} BB",
                                fill="#111111", font=("Arial", 13, "bold"))

        # Button states/status
        self.update_button_states()
        if self.env.done:
            self.update_status_end()
        elif self.env.current_player == self.player_idx:
            self.status_label.config(text=">> YOUR TURN <<")
        else:
            self.status_label.config(text="Bot's turn...")
    
    def update_button_states(self):
        """Enable/disable buttons based on game state."""
        if self.env.done:
            self.fold_btn.config(state='disabled')
            self.call_btn.config(state='disabled')
            self.raise_btn.config(state='disabled')
            return
        
        if self.env.current_player != self.player_idx:
            self.fold_btn.config(state='disabled')
            self.call_btn.config(state='disabled')
            self.raise_btn.config(state='disabled')
            return
        
        legal_actions = self.env._get_legal_actions()
        
        self.fold_btn.config(state='normal' if self.env.ACT_FOLD in legal_actions else 'disabled')
        self.call_btn.config(state='normal' if self.env.ACT_CHECK_CALL in legal_actions else 'disabled')
        self.raise_btn.config(state='normal' if any(a >= 2 for a in legal_actions) else 'disabled')
    
    def action_fold(self):
        """Handle fold action."""
        if self.env.done or self.env.current_player != self.player_idx:
            return
        
        self.status_label.config(text="You folded.")
        self.env.step(self.env.ACT_FOLD)
        self.redraw()
    
    def action_call(self):
        """Handle check/call action."""
        if self.env.done or self.env.current_player != self.player_idx:
            return
        
        to_call = self.env.get_to_call()
        if to_call > 0:
            self.status_label.config(text=f"You called {to_call:.1f} BB")
        else:
            self.status_label.config(text="You checked.")
        
        self.env.step(self.env.ACT_CHECK_CALL)
        self.redraw()
        
        # Bot's turn
        if not self.env.done:
            self.root.after(1000, self.bot_turn)
    
    def action_raise(self):
        """Handle raise action."""
        if self.env.done or self.env.current_player != self.player_idx:
            return
        
        self.status_label.config(text="You raised (1/2 pot)")
        self.env.step(self.env.ACT_RAISE_HALF_POT)
        self.redraw()
        
        # Bot's turn
        if not self.env.done:
            self.root.after(1000, self.bot_turn)
    
    def bot_turn(self):
        """Let bot take action."""
        if self.env.done or self.env.current_player != self.bot_idx:
            return
        
        # Get bot action
        obs = self.env._get_obs()
        action = self.bot.get_action(self.env, obs)
        
        action_names = {
            self.env.ACT_FOLD: "folded",
            self.env.ACT_CHECK_CALL: "called" if self.env.get_to_call() > 0 else "checked",
            self.env.ACT_RAISE_HALF_POT: "raised (1/2 pot)"
        }
        
        action_name = action_names.get(action, "acted")
        self.status_label.config(text=f"Bot {action_name}.")
        
        self.env.step(action)
        self.redraw()
        
        # If hand continues and it's player's turn, update status
        if not self.env.done and self.env.current_player == self.player_idx:
            self.status_label.config(text=">> YOUR TURN <<")
    
    def update_status_end(self):
        """Update status when hand ends."""
        if not hasattr(self.env, 'rewards') or self.env.rewards is None:
            return
        
        reward = self.env.rewards[self.player_idx]
        
        if reward > 0:
            self.status_label.config(text=f"🎉 You won {reward:.1f} BB!")
        elif reward < 0:
            self.status_label.config(text=f"😔 You lost {abs(reward):.1f} BB")
        else:
            self.status_label.config(text="🤝 Split pot")
        
        # Show bot's hand
        if len(self.env.hands) > self.bot_idx:
            bot_hand_str = ' '.join([self.card_to_str(c) for c in self.env.hands[self.bot_idx]])
            current_text = self.status_label.cget("text")
            self.status_label.config(text=f"{current_text} | Bot's hand: {bot_hand_str}")
    
    def new_hand(self):
        """Start a new hand."""
        self.env.reset()
        self.redraw()
        self.status_label.config(text="New hand started. Waiting for action...")
        
        # If bot is first to act, let it act
        if not self.env.done and self.env.current_player == self.bot_idx:
            self.root.after(1000, self.bot_turn)

    def _draw_card_row(self, center_x, center_y, cards, up=True, total_slots=None, card_width=60):
        """Draw a row of cards (face up or down)."""
        cards = list(cards) if cards else []
        if total_slots:
            # pad to total slots
            while len(cards) < total_slots:
                cards.append(None)
            cards = cards[:total_slots]

        spacing = card_width + 10
        start_x = center_x - (spacing * (len(cards) - 1)) / 2
        for i, c in enumerate(cards):
            x = start_x + i * spacing
            self._draw_card(x, center_y, c, up=up)

    def _draw_card(self, x, y, card, up=True):
        """Draw a single card at (x,y)."""
        w, h = 70, 96
        x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        fill = "#ffffff" if up else "#888888"
        outline = "#000000"
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, width=2)
        if up and card:
            txt = self.card_to_str(card)
            rank = txt[0]
            suit = txt[1]
            color = "red" if suit in ["h", "d"] else "black"
            self.canvas.create_text(x, y, text=f"{rank}{suit}", font=("Courier", 16, "bold"), fill=color)

def main():
    root = tk.Tk()
    app = HoldemPokerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
