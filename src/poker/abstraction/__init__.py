"""
Abstraction module for Poker CFR.

This module provides card and action abstraction to make CFR tractable
for large poker games like Texas Hold'em.

Key components:
- CardAbstraction: Buckets hands into clusters (169 preflop, ~25k postflop)
- ActionAbstraction: Simplifies bet sizing to discrete actions
"""

from .card_abstraction import CardAbstraction
from .action_abstraction import ActionAbstraction, Action

__all__ = [
    'CardAbstraction',
    'ActionAbstraction',
    'Action',
]
