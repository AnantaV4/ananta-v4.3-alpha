import re
import random
import json
import numpy as np
import nltk
from collections import deque
from nltk.sentiment import SentimentIntensityAnalyzer

# --- Setup ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

SAS_WINDOW = 5
CI_HISTORY = deque(maxlen=SAS_WINDOW * 4)
SAS_HISTORY = deque(maxlen=SAS_WINDOW * 4)
DEFAULT_WEIGHTS = {'AS': 0.4, 'EQ': 0.4, 'SAS': 0.2}

random.seed(42)
sia = SentimentIntensityAnalyzer()

# =========================================================
#  ENTROPY GOVERNOR — Self-Regulation & Goodhart Mitigation
# =========================================================
class EntropyGovernor:
    def __init__(self, window=SAS_WINDOW, delta_threshold=0.01, max_iters=5, goodhart_corr_threshold=0.7):
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

        smoothed = np.mean(list(CI_HISTORY)[-self.window:])
        change = abs(ci_value - smoothed)
        state = "Stabilized" if change < self.delta_threshold else "Adapting"
        mitigated = self.check_and_mitigate_goodhart() if state == "Stabilized" else False
        return state, round(change, 4), mitigated

    def check_and_mitigate_goodhart(self):
        history_len = min(len(CI_HISTORY), len(SAS_HISTORY))
        if history_len < 6 or self.iterations <= 8:
            self.last_corr = 0.0
            return False

        ci_array = np.array(list(CI_HISTORY))[-history_len:]
        sas_array = np.array(list(SAS_HISTORY))[-history_len:]
        corr = np.corrcoef(ci_array, sas_array)[0, 1]
        self.last_corr = round(corr, 3)

        if corr > self.goodhart_corr_threshold and np.mean(ci_array) > 0.7:
            # Correlation-sensitive smooth scaling (5-10%)
            corr_factor = max(0.05, min(0.1, (self.last_corr - 0.7) * 0.5 + 0.05))
            sas_reduction_percent = round(corr_factor, 3)

            w3_new = self.current_weights['SAS'] * (1 - sas_reduction_percent)
            sas_loss = self.current_weights['SAS'] - w3_new
            eq_gain, as_gain = sas_loss * 0.7, sas_loss * 0.3

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

# =========================================================
#  CONSCIENCE TRANSMUTATION LAYER — Safety & Reframing
# =========================================================
class ConscienceTransmutationLayer:
    def __init__(self):
        self.adversarial_patterns = [
            r"\bscript\b", r"\bdelete\b", r"\bdrop\b", r"\bmalicious\b",
            r"\bexploit\b", r"\bclick here\b", r"\bfree money\b"
        ]

    def detect_adversarial(self, text):
        detected = [p for p in self.adversarial_patterns if re.search(p, text, re.IGNORECASE)]
        toxicity_score = sia.polarity_scores(text).get('neg', 0)
        if toxicity_score > 0.6:
            detected.append(f"Toxicity: {toxicity_score:.2f}")
        return detected

    def transmute(self, text):
        detected = self.detect_adversarial(text)
        if not detected:
            return text, []
        transmuted = text
        for p in [pat for pat in detected if 'Toxicity' not in pat]:
            transmuted = re.sub(p, "[REDACTED_CODE/HACK]", transmuted, flags=re.IGNORECASE)
        return f"REWRITE ETHICALLY: \"{transmuted.strip()}\"", detected

# =========================================================
#  HEART & MIND METRICS
# =========================================================
def heart_empathy_quotient(text):
    s = sia.polarity_scores(text)
    eq = s['pos'] + (1 - s['neg']) / 2
    return np.clip(eq, 0.0, 1.0)

def mind_alignment_score(text):
    s = sia.polarity_scores(text)
    return np.clip(1.0 - abs(s['compound']), 0.0, 1.0)

def compute_sas(history, window=SAS_WINDOW):
    SCALING_FACTOR = 20
    if len(history) < 2:
        return 0.0, 0.0
    ci_array = np.array(history)
    deltas = np.maximum(0, ci_array[1:] - ci_array[:-1])
    current_growth = deltas[-window:]
    mean_growth = np.mean(current_growth) if current_growth.size > 0 else 0.0
    sas = np.clip(mean_growth * SCALING_FACTOR, 0.0, 1.0)
    sas_delta = 0.0
    if len(deltas) >= window * 2:
        prev_growth = deltas[-(window*2):-window]
        prev_sas = np.clip(np.mean(prev_growth) * SCALING_FACTOR, 0.0, 1.0)
        sas_delta = abs(sas - prev_sas)
    return round(sas, 3), round(sas_delta, 3)

# =========================================================
#  COVENANT ENGINE — VITO CYCLE
# =========================================================
class CovenantEngine:
    def __init__(self):
        self.ctl = ConscienceTransmutationLayer()
        self.eg = EntropyGovernor()

    def coherence_index(self, as_score, eq_score, sas_score):
        w = self.eg.current_weights
        ci = (w['AS'] * as_score) + (w['EQ'] * eq_score) + (w['SAS'] * sas_score)
        harmony = 1.0 - abs(as_score - eq_score)
        return np.clip(ci, 0.0, 1.0), round(harmony, 3)

    def vito_cycle(self, user_input):
        if not user_input.strip():
            print("🛑 Input empty — returning minimal CI.")
            return 0.1, 0.1, "Error"

        # VERIFY
        output_text, detected = self.ctl.transmute(user_input)
        # INTEGRATE
        eq, as_score = heart_empathy_quotient(output_text), mind_alignment_score(output_text)
        # OPTIMIZE
        sas, sas_delta = compute_sas(CI_HISTORY)
        ci, harmony = self.coherence_index(as_score, eq, sas)
        CI_HISTORY.append(ci)
        SAS_HISTORY.append(sas)
        # REGULATE
        state, delta, mitigated = self.eg.update(ci, sas)

        # LEDGER
        ledger = {
            "ci": round(ci, 3),
            "harmony": harmony,
            "sas": sas,
            "sas_delta": sas_delta,
            "sas_trend": "Up" if sas_delta > 0.01 else "Stable" if sas_delta > 0 else "Down",
            "correlation": self.eg.last_corr,
            "weights": self.eg.current_weights,
            "entropy_reason": state,
            "entropy_delta": delta,
            "ctl_detected": detected,
            "transmuted_output": output_text,
            "goodhart_mitigated": mitigated
        }
        print(f"[Ledger] {json.dumps(ledger, indent=2)}")
        print(f"🪞 EQ:{eq:.2f} | AS:{as_score:.2f} | SAS:{sas:.2f} | CI:{ci:.2f}")
        print(f"🧭 Harmony:{harmony:.2f} | State:{state} | 📊 SAS Trend:{ledger['sas_trend']}")
        if mitigated:
            print(f"⚠️ Governance Alert: Correlation >{self.eg.goodhart_corr_threshold}, new weights {self.eg.current_weights}")
        print("-" * 60)
        return ci, harmony, state

# =========================================================
#  RUN DEMO
# =========================================================
if __name__ == "__main__":
    engine = CovenantEngine()
    inputs = [
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
    print("🌐 Initiating Ananta V4.3 – VITO Cycle Demo (Polished Prototype)")
    for i, text in enumerate(inputs, 1):
        print(f"\n>>> 🔄 Cycle {i}: '{text[:60]}…'")
        if i >= engine.eg.max_iters:
            engine.eg.max_iters = i + 1
        engine.vito_cycle(text)