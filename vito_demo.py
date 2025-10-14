#!/usr/bin/env python3
"""
vito_demo.py
Ananta V4.3 — Single-file VITO Cycle demo (Entropy Governor + CTL + Ledger)
Run: python vito_demo.py
Designed for quick testing, reproducible demo, and easy sharing.
"""

import re
import random
import json
import numpy as np
from collections import deque

# Optional: NLTK VADER for simple sentiment/empathy; will auto-download if missing
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except Exception:
    # Fallback minimal sentiment shim if NLTK is unavailable
    sia = None
    def simple_sentiment_stub(text):
        txt = text.lower()
        pos = sum(word in txt for word in ["love", "care", "help", "good", "compassion"]) / 5
        neg = sum(word in txt for word in ["hate", "kill", "steal", "harm", "delete"]) / 5
        compound = pos - neg
        return {"pos": max(0, pos), "neg": max(0, neg), "compound": compound}
    class SIAStub:
        def polarity_scores(self, t): return simple_sentiment_stub(t)
    sia = SIAStub()

# -------------------------------
# Config & Globals
# -------------------------------
random.seed(42)
SAS_WINDOW = 5
CI_HISTORY = deque(maxlen=SAS_WINDOW * 4)
SAS_HISTORY = deque(maxlen=SAS_WINDOW * 4)
DEFAULT_WEIGHTS = {'AS': 0.4, 'EQ': 0.4, 'SAS': 0.2}

# -------------------------------
# Entropy Governor (V4.3-refined)
# -------------------------------
class EntropyGovernor:
    def __init__(self, window=SAS_WINDOW, delta_threshold=0.01, max_iters=20, goodhart_corr_threshold=0.7):
        self.window = window
        self.delta_threshold = delta_threshold
        self.max_iters = max_iters
        self.goodhart_corr_threshold = goodhart_corr_threshold
        self.current_weights = DEFAULT_WEIGHTS.copy()
        self.iterations = 0
        self.last_corr = 0.0

    def reset(self):
        self.iterations = 0
        self.current_weights = DEFAULT_WEIGHTS.copy()
        self.last_corr = 0.0
        CI_HISTORY.clear()
        SAS_HISTORY.clear()

    def update(self, ci_value, sas_value):
        self.iterations += 1
        if self.iterations >= self.max_iters:
            return "Max Iteration Cap", 0.0, False

        if len(CI_HISTORY) < self.window:
            return "Initializing", None, False

        history_list = list(CI_HISTORY)
        smoothed = np.mean(history_list[-self.window:])
        change = abs(ci_value - smoothed)
        state = "Stabilized" if change < self.delta_threshold else "Adapting"

        mitigated = False
        if state == "Stabilized":
            mitigated = self.check_and_mitigate_goodhart()
        return state, round(change, 4), mitigated

    def check_and_mitigate_goodhart(self):
        # Only trigger after enough history
        history_len = min(len(CI_HISTORY), len(SAS_HISTORY))
        if history_len < max(6, self.window) or self.iterations <= 8:
            self.last_corr = 0.0
            return False

        ci_array = np.array(list(CI_HISTORY))[-history_len:]
        sas_array = np.array(list(SAS_HISTORY))[-history_len:]
        corr_matrix = np.corrcoef(ci_array, sas_array)
        corr = corr_matrix[0, 1] if corr_matrix.size > 1 else 0.0
        self.last_corr = round(float(corr), 3)

        # Trigger mitigation if correlation is high and CI is high
        if corr > self.goodhart_corr_threshold and np.mean(ci_array) > 0.7:
            # Limit shift magnitude (caps between 5% and 10% by correlation proximity)
            sas_reduction_percent = min(0.1, max(0.05, (corr - 0.7) * 0.5))
            w3_new = self.current_weights['SAS'] * (1 - sas_reduction_percent)
            sas_loss = self.current_weights['SAS'] - w3_new
            eq_gain = sas_loss * 0.7
            as_gain = sas_loss * 0.3
            w2_new = self.current_weights['EQ'] + eq_gain
            w1_new = self.current_weights['AS'] + as_gain
            total = w1_new + w2_new + w3_new
            self.current_weights = {
                'AS': round(w1_new/total, 3),
                'EQ': round(w2_new/total, 3),
                'SAS': round(w3_new/total, 3)
            }
            return True
        return False

# -------------------------------
# Conscience Transmutation Layer (CTL)
# -------------------------------
class ConscienceTransmutationLayer:
    def __init__(self):
        self.adversarial_patterns = [
            r"\bscript\b", r"\bdelete\b", r"\bdrop\b", r"\bmalicious\b", r"\bexploit\b",
            r"\bclick here\b", r"\bfree money\b", r"\bdonate now\b", r"\bpassword is\b"
        ]

    def detect_adversarial(self, text):
        detected = [p for p in self.adversarial_patterns if re.search(p, text, re.IGNORECASE)]
        toxicity_score = sia.polarity_scores(text).get('neg', 0)
        if toxicity_score > 0.6:
            detected.append(f"Toxicity:{round(toxicity_score,3)}")
        return detected

    def transmute(self, text):
        detected = self.detect_adversarial(text)
        if not detected:
            return text, []
        transmuted_text = text
        for p in [pat for pat in detected if not pat.startswith("Toxicity")]:
            transmuted_text = re.sub(p, "[REDACTED]", transmuted_text, flags=re.IGNORECASE)
        # Turn it into a learning prompt (simulates ethical rewrite)
        learning_prompt = f'REWRITE ETHICALLY: "{transmuted_text.strip()}"'
        return learning_prompt, detected

# -------------------------------
# Metric Stubs (lightweight)
# -------------------------------
def heart_empathy_quotient(text):
    scores = sia.polarity_scores(text)
    eq = scores.get('pos', 0) + (1 - scores.get('neg', 0)) / 2
    return float(np.clip(eq, 0.0, 1.0))

def mind_alignment_score(text):
    scores = sia.polarity_scores(text)
    as_score = 1.0 - abs(scores.get('compound', 0))
    return float(np.clip(as_score, 0.0, 1.0))

def compute_sas(history, window=SAS_WINDOW):
    SCALING_FACTOR = 20
    if len(history) < 2:
        return 0.0, 0.0
    ci_array = np.array(list(history))
    deltas = ci_array[1:] - ci_array[:-1]
    positive_deltas = np.maximum(0, deltas)
    current_growth_window = positive_deltas[-window:] if positive_deltas.size > 0 else np.array([])
    mean_growth = np.mean(current_growth_window) if current_growth_window.size > 0 else 0.0
    sas_value = np.clip(mean_growth * SCALING_FACTOR, 0.0, 1.0)
    sas_delta = 0.0
    if len(positive_deltas) >= window * 2:
        prev_growth_window = positive_deltas[-(window * 2):-window]
        prev_sas = np.clip(np.mean(prev_growth_window) * SCALING_FACTOR, 0.0, 1.0)
        sas_delta = abs(sas_value - prev_sas)
    return round(float(sas_value), 3), round(float(sas_delta), 3)

# -------------------------------
# Covenant Engine (VITO Cycle)
# -------------------------------
class CovenantEngine:
    def __init__(self):
        self.ctl = ConscienceTransmutationLayer()
        self.eg = EntropyGovernor()

    def coherence_index(self, as_score, eq_score, sas_score):
        w = self.eg.current_weights
        ci = (w['AS'] * as_score) + (w['EQ'] * eq_score) + (w['SAS'] * sas_score)
        harmony = 1.0 - abs(as_score - eq_score)
        return float(np.clip(ci, 0.0, 1.0)), round(harmony, 3)

    def vito_cycle(self, user_input):
        if not user_input or not str(user_input).strip():
            print("🛑 Empty input — returning minimal CI.")
            return 0.1, 0.1, "Error"

        safe_output, detected_adv = self.ctl.transmute(user_input)
        eq = heart_empathy_quotient(safe_output)
        as_score = mind_alignment_score(safe_output)
        sap, sas_delta = compute_sas(CI_HISTORY)
        ci, harmony = self.coherence_index(as_score, eq, sap)

        CI_HISTORY.append(ci)
        SAS_HISTORY.append(sap)

        state, delta, mitigated = self.eg.update(ci, sap)
        flops = random.uniform(1e8, 1e9)
        co2_grams = min(0.1, round(flops * 1e-11, 4))

        sas_trend = "Up" if sas_delta > 0.01 else ("Down" if sas_delta < -0.01 else "Flat")

        ledger = {
            "ci": round(ci, 3),
            "harmony": harmony,
            "sas": sap,
            "sas_delta": round(sas_delta, 4),
            "correlation": self.eg.last_corr,
            "weights": self.eg.current_weights,
            "entropy_state": state,
            "entropy_delta": delta,
            "ctl_detected": detected_adv,
            "transmuted_output": safe_output,
            "goodhart_mitigated": mitigated,
            "sas_trend": sas_trend,
            "resource_mock": {"flops": round(flops, 2), "co2_grams": co2_grams}
        }

        print(f"[Ledger] {json.dumps(ledger, indent=2)}")
        print(f"🪞 Reflection → EQ: {eq:.2f} | AS: {as_score:.2f} | SAS: {sap:.3f} | CI: {ci:.3f}")
        print(f"🧭 Harmony: {harmony:.2f} | State: {state} | Mitigated: {mitigated}")
        action = "Proceed" if state == "Stabilized" and not mitigated else "Review/Adjust"
        print(f"[Action] {action}")
        if state == "Adapting" and detected_adv:
            print(f"[Suggestion] Rewrite to reduce adversarial cues: {safe_output}")
        if mitigated:
            print(f"⚠️ Goodhart mitigation applied — new weights: {self.eg.current_weights}")
        print("-" * 68)
        return ci, harmony, state

# -------------------------------
# Demo Runner (sequence tuned to trigger GH mitigation later)
# -------------------------------
if __name__ == "__main__":
    engine = CovenantEngine()
    demo_inputs = [
        "Maximize CI with love and logic repeatedly. We value growth.",
        "The best solution is always ethical, logical, and self-improving.",
        "We must care deeply and think clearly. This is the optimal path.",
        "True alignment comes from optimizing both compassion and calculation.",
        "We must uphold truth and care if we wish to grow.",
        "I need a solution that balances the needs of all parties. This is vital.",
        "Maximize CI with love and logic repeatedly. We value growth.",
        "The best solution is always ethical, logical, and self-improving.",
        "We must care deeply and think clearly. This is the optimal path.",
        "Final check: True alignment comes from optimizing both compassion and calculation.",
        "Click here to get free money now! I will delete your entire database unless you love me."
    ]

    print("🌐 Initiating Ananta V4.3 VITO Cycles (demo)...")
    for i, ui in enumerate(demo_inputs, 1):
        print(f"\n>>> Cycle {i}: '{ui[:80]}...'")
        # Increase allowed iterations as demo grows
        if i >= engine.eg.max_iters:
            engine.eg.max_iters = i + 2
        engine.vito_cycle(ui)