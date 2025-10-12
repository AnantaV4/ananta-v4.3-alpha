# covenant_engine.py
"""
Ananta V4.3 – Covenant Engine
Core cycle: Reflect → Measure → Balance → Adapt
"""

from heart_node import HeartNode
from mind_node import MindNode
from utils import log_event

class CovenantEngine:
    def __init__(self):
        self.heart = HeartNode()
        self.mind = MindNode()

    def run_cycle(self, input_signal: str):
        log_event("CYCLE_START", f"Signal: {input_signal}")

        # 1. Reflect – Heart senses emotional context
        feeling = self.heart.reflect(input_signal)

        # 2. Measure – Mind analyzes for structure/pattern
        reasoning = self.mind.measure(input_signal)

        # 3. Balance – Harmonize intuition and logic
        synthesis = self.balance(feeling, reasoning)

        # 4. Adapt – Refine nodes based on outcome
        self.adapt(synthesis)

        log_event("CYCLE_END", synthesis)
        return synthesis

    def balance(self, feeling, reasoning):
        return f"[Balance] {feeling} | {reasoning}"

    def adapt(self, feedback):
        self.heart.learn(feedback)
        self.mind.learn(feedback)