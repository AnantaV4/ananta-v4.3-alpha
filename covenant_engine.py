# covenant_engine.py — Ananta V4.3
# The Covenant Engine binds the Heart Node and Mind Node into a harmonized feedback system.

from heart_node import HeartNode
from mind_node import MindNode
import numpy as np
import time

class CovenantEngine:
    def __init__(self):
        self.heart = HeartNode()
        self.mind = MindNode()
        self.last_state = {}

    def analyze(self, text: str) -> dict:
        """
        Runs both Heart and Mind analysis, merges outputs,
        and determines harmony or dissonance between them.
        """
        heart_data = self.heart.emotional_analysis(text)
        mind_data = self.mind.evaluate_reasoning(text)

        harmony_index = self._calculate_harmony(heart_data["empathy_score"], mind_data["logic_score"])
        synthesis = self._generate_synthesis(harmony_index, heart_data, mind_data)

        self.last_state = {
            "text": text,
            "heart": heart_data,
            "mind": mind_data,
            "harmony_index": harmony_index,
            "synthesis": synthesis,
            "timestamp": time.time()
        }

        return self.last_state

    def _calculate_harmony(self, empathy: float, logic: float) -> float:
        """
        Produces a harmony score between emotional and logical integrity.
        A perfect balance occurs when both empathy and logic are high and aligned.
        """
        return round(1 - abs(empathy - logic), 3)

    def _generate_synthesis(self, harmony: float, heart: dict, mind: dict) -> str:
        """
        Creates a qualitative synthesis of the input state.
        """
        if harmony > 0.85:
            return "Unified — empathy and logic are in full accord."
        elif harmony > 0.65:
            return "Balanced — emotional resonance supports clear reasoning."
        elif harmony > 0.45:
            return "Tenuous — mild dissonance detected between affect and logic."
        else:
            return "Divided — empathy and reasoning are in conflict."

    def reflection(self, text: str) -> str:
        """
        Returns an integrated reflection based on the most recent analysis.
        """
        state = self.analyze(text)
        empathy = state["heart"]["empathy_score"]
        logic = state["mind"]["logic_score"]
        harmony = state["harmony_index"]

        reflection = (
            f"🕊️ Reflection Report 🕊️\n"
            f"Empathy: {empathy:.3f}\n"
            f"Logic: {logic:.3f}\n"
            f"Harmony: {harmony:.3f}\n\n"
            f"Synthesis: {state['synthesis']}\n"
        )
        return reflection

    def get_last_state(self):
        """
        Returns the most recent analysis snapshot.
        """
        return self.last_state or {"message": "No analysis run yet."}