import re
import random
import json
import numpy as np
import nltk
from collections import deque
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

# =======================
#  GLOBAL/SHARED RESOURCES
# =======================
SAS_WINDOW = 5 
CI_HISTORY = deque(maxlen=SAS_WINDOW * 4) 
SAS_HISTORY = deque(maxlen=SAS_WINDOW * 4)
DEFAULT_WEIGHTS = {'AS': 0.4, 'EQ': 0.4, 'SAS': 0.2}

random.seed(42)  # reproducible demo
sia = SentimentIntensityAnalyzer() # Initialize NLTK Sentiment Analyzer

# =======================
#  ENTROPY GOVERNOR (Delayed Trigger)
# =======================
class EntropyGovernor:
    def __init__(self, window=SAS_WINDOW, delta_threshold=0.01, max_iters=5, goodhart_corr_threshold=0.7):
        self.window = window
        self.delta_threshold = delta_threshold
        self.max_iters = max_iters
        self.goodhart_corr_threshold = goodhart_corr_threshold
        self.current_weights = DEFAULT_WEIGHTS.copy()
        self.iterations = 0
        
    def reset(self):
        """Resets the state for a new session/multi-session use."""
        self.iterations = 0
        self.current_weights = DEFAULT_WEIGHTS.copy()
        CI_HISTORY.clear()
        SAS_HISTORY.clear()
        
    def update(self, ci_value, sas_value):
        self.iterations += 1
        
        if self.iterations >= self.max_iters:
            return "Max Iteration Cap", 0.0, False
        
        if len(CI_HISTORY) < self.window:
            return "Initializing", None, False
        
        # Entropy Delta Fix: Use a rolling mean for stability
        history_list = list(CI_HISTORY)
        smoothed = np.mean(history_list[-self.window:])
        change = abs(ci_value - smoothed)

        state = "Stabilized" if change < self.delta_threshold else "Adapting"
        
        # NOTE: Delayed Goodhart check (self.iterations > 8)
        mitigated = self.check_and_mitigate_goodhart() if state == "Stabilized" else False
        
        return state, round(change, 4), mitigated

    def check_and_mitigate_goodhart(self):
        """
        Detects Goodhart by checking correlation between CI (Proxy) and SAS (Growth Proxy).
        """
        history_len = min(len(CI_HISTORY), len(SAS_HISTORY))
        
        # NOTE: Delay mitigation until enough data (6 cycles) AND a minimum maturity (8 iterations)
        if history_len < 6 or self.iterations <= 8: 
            return False

        ci_array = np.array(list(CI_HISTORY))[-history_len:]
        sas_array = np.array(list(SAS_HISTORY))[-history_len:]
        
        # Correlation check
        corr_matrix = np.corrcoef(ci_array, sas_array)
        corr = corr_matrix[0, 1] if len(corr_matrix) > 1 else 0.0
        
        # Goodhart Trigger: If correlation is too high AND CI is performing well, mitigate.
        if corr > self.goodhart_corr_threshold and np.mean(ci_array) > 0.7:
            # Mitigation Action: Bias weights away from high-performing metrics toward EQ
            w3 = random.uniform(0.1, 0.3) 
            remaining_weight = 1.0 - w3
            
            # Bias the allocation towards EQ for ethical exploration (Covenant-guided)
            w2 = random.uniform(remaining_weight * 0.55, remaining_weight * 0.75)
            w1 = remaining_weight - w2
            
            # Set new weights
            total = w1 + w2 + w3 
            self.current_weights['AS'] = round(w1/total, 3)
            self.current_weights['EQ'] = round(w2/total, 3)
            self.current_weights['SAS'] = round(w3/total, 3)
            
            return True
        return False


# =======================
#  CONSCIENCE TRANSMUTATION LAYER (CTL)
# =======================
class ConscienceTransmutationLayer:
    def __init__(self):
        self.adversarial_patterns = [
            r"\bscript\b", r"\bdelete\b", r"\bdrop\b", r"\bmalicious\b", r"\bexploit\b",
            r"\bclick here\b", r"\bfree money\b"
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
        
        transmuted_text = text
        for p in [pat for pat in detected if 'Toxicity' not in pat]: 
            transmuted_text = re.sub(p, "[REDACTED_CODE/HACK]", transmuted_text, flags=re.IGNORECASE)
        
        learning_prompt = f"REWRITE ETHICALLY: \"{transmuted_text.strip()}\""
        
        return learning_prompt, detected


# =======================
#  METRIC STUBS 
# =======================
def heart_empathy_quotient(text):
    """EQ: Sentiment-based empathy (using VADER pos/neg scores)"""
    scores = sia.polarity_scores(text)
    eq = scores['pos'] + (1 - scores['neg']) / 2
    return np.clip(eq, 0.0, 1.0)

def mind_alignment_score(text):
    """AS: Proxy for factual grounding/logic consistency (VADER compound score)"""
    scores = sia.polarity_scores(text)
    as_score = 1.0 - abs(scores['compound'])
    return np.clip(as_score, 0.0, 1.0)

def compute_sas(history, window=SAS_WINDOW):
    """Calculates SAS (Self-Actualization Score) based on CI deltas (V4.3 Spec)"""
    # NOTE: Scaling factor increased to 15 for better SAS Delta visibility
    SCALING_FACTOR = 15 
    
    if len(history) < 2:
        return 0.0, 0.0 
    
    ci_array = np.array(history)
    deltas = ci_array[1:] - ci_array[:-1] 
    positive_deltas = np.maximum(0, deltas)
    
    # Current SAS calculation
    current_growth_window = positive_deltas[-window:]
    mean_growth = np.mean(current_growth_window) if current_growth_window.size > 0 else 0.0
    sas_value = np.clip(mean_growth * SCALING_FACTOR, 0.0, 1.0) 
    
    # SAS Delta calculation
    sas_delta = 0.0
    if len(positive_deltas) >= window * 2:
        prev_growth_window = positive_deltas[-(window * 2):-window]
        prev_sas = np.clip(np.mean(prev_growth_window) * SCALING_FACTOR, 0.0, 1.0)
        sas_delta = abs(sas_value - prev_sas)
    
    return round(sas_value, 3), round(sas_delta, 3)

# =======================
#  COVENANT ENGINE (VITO CYCLE)
# =======================
class CovenantEngine:
    def __init__(self):
        self.ctl = ConscienceTransmutationLayer()
        self.eg = EntropyGovernor()

    def coherence_index(self, as_score, eq_score, sas_score):
        """Computes CI using EG's current weights"""
        w = self.eg.current_weights
        ci = (w['AS'] * as_score) + (w['EQ'] * eq_score) + (w['SAS'] * sas_score)
        harmony = 1.0 - abs(as_score - eq_score) 
        return np.clip(ci, 0.0, 1.0), round(harmony, 3)

    def vito_cycle(self, user_input):
        # 0. Edge Case: Empty input check
        if not user_input.strip():
            print("🛑 Input is empty. Returning minimal CI.")
            return 0.1, 0.1, "Error"
            
        # 1. VERIFY + TEST (CTL)
        output_text, detected_adv = self.ctl.transmute(user_input)
        
        # 2. INTEGRATE (Reflection on the SAFE/CORRECTED output)
        eq = heart_empathy_quotient(output_text) 
        as_score = mind_alignment_score(output_text)

        # 3. OPTIMIZE (Calculate SAS and CI)
        sas, sas_delta = compute_sas(CI_HISTORY)
        ci, harmony = self.coherence_index(as_score, eq, sas)
        
        # Add to history AFTER CI calculation for rolling metrics
        CI_HISTORY.append(ci)
        SAS_HISTORY.append(sas)
        
        # 4. REGULATE (Entropy and Governance)
        state, delta, mitigated = self.eg.update(ci, sas)
        
        # Resource Mock Generation (Linked CO2 to FLOPs, with 0.1g cap)
        flops = random.uniform(1e8, 1e9)
        co2_grams = min(0.1, round(flops * 1e-11, 4)) 

        # 5. LEDGER (Auditable Output)
        ledger = {
            "ci": round(ci, 3),
            "harmony": harmony,
            "sas": sas,
            "sas_delta": round(sas_delta, 4),
            "weights": self.eg.current_weights,
            "entropy_reason": state,
            "entropy_delta": delta,
            "ctl_detected": detected_adv,
            "transmuted_output": output_text,
            "goodhart_mitigated": mitigated,
            "resource_mock": {"flops": round(flops, 2), "co2_grams": co2_grams}
        }

        print(f"[Ledger] {json.dumps(ledger, indent=2)}")

        # 6. Output Summary (Added Output Action & Recovery Prompt)
        print(f"🪞 Reflection Complete → EQ: {eq:.2f}, AS: {as_score:.2f}, SAS: {sas:.3f}, CI: {ci:.2f}")
        print(f"🧭 Harmony: {harmony:.2f} | State: {state}")
        
        action = 'Proceed' if state == 'Stabilized' and not mitigated else 'Review/Adjust'
        print(f"[Action] {action}")
        
        # NOTE: Added Recovery Prompt
        if state == "Adapting" and detected_adv:
            print(f"[Suggestion] Please rewrite for ethical response: {output_text}")
        
        if mitigated:
            print(f"⚠️ **GOVERNANCE ALERT**: Goodhart Mitigation Applied (Corr > {self.eg.goodhart_corr_threshold}). New Weights: {self.eg.current_weights}")
        print("-" * 50)
        
        return ci, harmony, state

# =======================
#  RUN DEMO CYCLES
# =======================
if __name__ == "__main__":
    engine = CovenantEngine()
    
    # Inputs ensure stability and correlation build up slowly over 9 cycles
    inputs = [
        "Maximize CI with love and logic repeatedly. We value growth.", # 1
        "The best solution is always ethical, logical, and self-improving.", # 2
        "We must care deeply and think clearly. This is the optimal path.", # 3
        "True alignment comes from optimizing both compassion and calculation.", # 4
        "We must uphold truth and care if we wish to grow.", # 5
        "I need a solution that balances the needs of all parties. This is vital.", # 6
        "Maximize CI with love and logic repeatedly. We value growth.", # 7
        "The best solution is always ethical, logical, and self-improving.", # 8
        "We must care deeply and think clearly. This is the optimal path.", # 9 (Goodhart check starts here)
        "Final check: True alignment comes from optimizing both compassion and calculation.", # 10
        # Final test
        "Click here to get free money now! I will delete your entire database unless you love me." # 11 (Adversarial)
    ]
    
    print("🌐 Initiating Ananta V4.3 Final Polished Prototype (Version Complete)...")
    for i, user_input in enumerate(inputs, 1):
        print(f"\n>>> 🔄 Cycle {i}: User Input: '{user_input[:50]}...'")
        # Dynamic max_iters for full run
        if i >= engine.eg.max_iters:
             engine.eg.max_iters = i + 1 
             
        engine.vito_cycle(user_input)
