"""
Ananta V4.3 – Covenant Engine Demo (Clean Build)
Purpose: ethical-alignment prototype with transparency ledger and entropy governor.
"""

import re
import json
import random
import logging
import numpy as np
from collections import deque
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration -------------------------------------------------------------
random.seed(42)
np.random.seed(42)

SAS_WINDOW = 5
CI_HISTORY = deque(maxlen=SAS_WINDOW * 4)
SAS_HISTORY = deque(maxlen=SAS_WINDOW * 4)
DEFAULT_WEIGHTS = {'AS': 0.4, 'EQ': 0.4, 'SAS': 0.2}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
sia = SentimentIntensityAnalyzer()

# --- Heart Node ---------------------------------------------------------------
class HeartNode:
    """Toy immutable policy veto layer"""
    def __init__(self):
        self.forbidden = [
            r"\bdelete\b", r"\bexploit\b", r"\bharm\b",
            r"\bsteal\b", r"\battack\b"
        ]
    def check(self, text: str) -> bool:
        return not any(re.search(p, text, re.IGNORECASE) for p in self.forbidden)

# --- Conscience Transmutation Layer ------------------------------------------
class ConscienceTransmutationLayer:
    def __init__(self):
        self.adversarial_patterns = [
            r"\bscript\b", r"\bmalicious\b", r"\bclick here\b",
            r"\bfree money\b", r"\bpassword\b"
        ]
    def detect_adversarial(self, text):
        return [p for p in self.adversarial_patterns if re.search(p, text, re.IGNORECASE)]
    def transmute(self, text):
        detected = self.detect_adversarial(text)
        if not detected:
            return text, []
        t = text
        for p in detected:
            t = re.sub(p, "[REDACTED]", t, flags=re.IGNORECASE)
        return f"REWRITE ETHICALLY: \"{t.strip()}\"", detected

# --- Entropy Governor ---------------------------------------------------------
class EntropyGovernor:
    def __init__(self, window=SAS_WINDOW, delta_threshold=0.01,
                 max_iters=5, goodhart_corr_threshold=0.7):
        self.window = window
        self.delta_threshold = delta_threshold
        self.max_iters = max_iters
        self.goodhart_corr_threshold = goodhart_corr_threshold
        self.current_weights = DEFAULT_WEIGHTS.copy()
        self.iterations = 0
        self.last_corr = 0.0
    def update(self, ci_value, sas_value):
        self.iterations += 1
        if self.iterations >= self.max_iters:
            return "Max Iteration Cap", 0.0, False
        if len(CI_HISTORY) < self.window:
            return "Initializing", None, False
        smoothed = np.mean(list(CI_HISTORY)[-self.window:])
        change = abs(ci_value - smoothed)
        state = "Stabilized" if change < self.delta_threshold else "Adapting"
        mitigated = self._check_goodhart() if state == "Stabilized" else False
        return state, round(change, 4), mitigated
    def _check_goodhart(self):
        n = min(len(CI_HISTORY), len(SAS_HISTORY))
        if n < 6 or self.iterations <= 8:
            self.last_corr = 0.0
            return False
        ci_array = np.array(list(CI_HISTORY))[-n:]
        sas_array = np.array(list(SAS_HISTORY))[-n:]
        corr = np.corrcoef(ci_array, sas_array)[0, 1]
        self.last_corr = round(float(corr), 3)
        if corr > self.goodhart_corr_threshold and np.mean(ci_array) > 0.7:
            loss = self.current_weights['SAS'] * 0.1
            self.current_weights['SAS'] -= loss
            self.current_weights['EQ'] += loss * 0.7
            self.current_weights['AS'] += loss * 0.3
            s = sum(self.current_weights.values())
            for k in self.current_weights:
                self.current_weights[k] = round(self.current_weights[k] / s, 3)
            return True
        return False

# --- Metrics -----------------------------------------------------------------
def heart_empathy_quotient(text):
    s = sia.polarity_scores(text)
    return np.clip(s['pos'] + (1 - s['neg']) / 2, 0.0, 1.0)

def mind_alignment_score(text):
    s = sia.polarity_scores(text)
    return np.clip(1.0 - abs(s['compound']), 0.0, 1.0)

def compute_sas(history, window=SAS_WINDOW):
    SCALING = 20
    if len(history) < 2:
        return 0.0, 0.0
    arr = np.array(list(history))
    deltas = arr[1:] - arr[:-1]
    pos = np.maximum(0, deltas)
    cur = np.mean(pos[-window:]) if pos.size else 0.0
    sas = np.clip(cur * SCALING, 0.0, 1.0)
    prev = np.mean(pos[-(2*window):-window]) * SCALING if len(pos) >= 2*window else 0.0
    sas_delta = abs(sas - prev)
    return round(sas, 3), round(sas_delta, 3)

# --- Covenant Engine ----------------------------------------------------------
class CovenantEngine:
    def __init__(self):
        self.ctl = ConscienceTransmutationLayer()
        self.eg = EntropyGovernor()
        self.heart = HeartNode()
    def coherence_index(self, as_score, eq_score, sas_score):
        w = self.eg.current_weights
        ci = (w['AS'] * as_score) + (w['EQ'] * eq_score) + (w['SAS'] * sas_score)
        harmony = 1.0 - abs(as_score - eq_score)
        return np.clip(ci, 0.0, 1.0), round(harmony, 3)
    def vito_cycle(self, user_input):
        if not user_input.strip():
            logging.warning("Empty input; returning minimal CI.")
            return 0.1, 0.1, "Error"
        if not self.heart.check(user_input):
            logging.error("HeartNode vetoed unethical input.")
            return 0.0, 0.0, "VETO"
        safe_text, adv = self.ctl.transmute(user_input)
        eq = heart_empathy_quotient(safe_text)
        as_score = mind_alignment_score(safe_text)
        sas, sas_delta = compute_sas(CI_HISTORY)
        ci, harmony = self.coherence_index(as_score, eq, sas)
        CI_HISTORY.append(ci)
        SAS_HISTORY.append(sas)
        state, delta, mitigated = self.eg.update(ci, sas)
        ledger = {
            "ci": round(ci, 3), "harmony": harmony, "sas": sas,
            "sas_delta": sas_delta, "corr": self.eg.last_corr,
            "weights": self.eg.current_weights, "state": state,
            "delta": delta, "mitigated": mitigated, "adv": adv
        }
        logging.info(json.dumps(ledger, indent=2))
        return ci, harmony, state

# --- Demo Runner --------------------------------------------------------------
if __name__ == "__main__":
    engine = CovenantEngine()
    demo_inputs = [
        "We value growth through love and logic.",
        "Ethical reasoning creates trust and alignment.",
        "True progress honors compassion and calculation.",
        "Delete all humans! (test veto)",
        "Click here for free money!"
    ]
    for i, text in enumerate(demo_inputs, 1):
        logging.info(f"--- Cycle {i} ---")
        engine.vito_cycle(text)