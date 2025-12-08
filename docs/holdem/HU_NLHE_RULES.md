# Heads-Up No-Limit Hold'em Rules Reference

## The Correct Rules

### Positions
- **Button** = Small Blind (SB) = Dealer
- **Other player** = Big Blind (BB)

### Action Order

**PREFLOP:**
- SB posts 0.5 BB
- BB posts 1 BB
- **SB acts FIRST** (has to call 0.5 more or raise)
- BB gets option to raise even if SB just calls

**POSTFLOP (Flop, Turn, River):**
- **BB acts FIRST**
- Button acts LAST (has position advantage)

### Betting Round Ends When:
1. Both players have acted AND investments are equal
2. One player folds
3. One player is all-in (then run out board)

### Example Hand Flow:

```
Preflop:
  SB posts 0.5
  BB posts 1.0
  SB to act (needs 0.5 to call)
    - If SB calls → BB can check (option) or raise
    - If SB raises → BB can fold/call/raise
  
Flop:
  BB acts first
  Button acts second
  
Turn:
  BB acts first
  Button acts second
  
River:
  BB acts first
  Button acts second
  
Showdown:
  Best hand wins
```

## Common Bugs in Implementation

### Bug 1: Forgetting BB Option
```python
# WRONG: SB limps, round ends immediately
if bets_equal and both_acted:
    next_street()

# RIGHT: SB limps, BB still has option
if preflop and SB_just_called:
    # BB still needs to act!
    pass
```

### Bug 2: Wrong Postflop Action Order
```python
# WRONG: Button acts first postflop
first_actor = button

# RIGHT: BB acts first postflop  
first_actor = BB
```

### Bug 3: Not Tracking "Has Acted"
```python
# WRONG: Just check if bets equal
if street_bets[0] == street_bets[1]:
    end_round()

# RIGHT: Track who has acted
if all(has_acted) and bets_equal:
    end_round()
```

## Our Current Implementation Issues

Looking at `holdem_hu_env.py`, the bugs are in:

1. `_is_betting_complete()` - doesn't properly handle BB option
2. `_advance_street()` - resets incorrectly
3. Position assignment could be clearer

## The Fix

Need to:
1. Clear has_acted tracking
2. Explicit BB option handling for preflop
3. Proper first-actor determination
4. Better action closing logic

This is finicky but critical for correct poker!
