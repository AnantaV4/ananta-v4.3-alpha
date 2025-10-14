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
# Using deque for efficient rolling window history (V4.3 persistence refinement)
CI_HISTORY = deque(maxlen=20) # Max 20 history points for CI stabilization + SAS calculation
SAS_WINDOW = 3 # Window for SAS calculation
DEFAULT_WEIGHTS = {'AS': 0.4, 'EQ': 0.4, 'SAS': 0.2}

random.seed(42)  # reproducible demo
sia = SentimentIntensityAnalyzer() # Initialize NLTK Sentiment Analyzer

# =======================
#  ENTROPY GOVERNOR (Refined with Covenant-Guided Mitigation)
# =======================
class EntropyGovernor:
    def __init__(self, window=3, delta_threshold=0.01, max_iters=5, goodhart_corr_threshold=0.9):
        self.window = window
        self.delta_threshold = delta_threshold
        self.max_iters = max_iters
        self.goodhart_corr_threshold = goodhart_corr_threshold
        self.current_weights = DEFAULT_WEIGHTS.copy()
        self.iterations = 0
        self.sas_history = [] # To track SAS for correlation check
        
    def reset(self):
        """Resets the state for a new session/multi-session use."""
        self.iterations = 0
        self.current_weights = DEFAULT_WEIGHTS.copy()
        self.sas_history = []
        
    def update(self, ci_value, sas_value):
        self.iterations += 1
        self.sas_history.append(sas_value)
        
        # 1. Termination Check
        if self.iterations >= self.max_iters:
            return "Max Iteration Cap", 0.0, False
        if len(CI_HISTORY) < self.window:
            return "Initializing", None, False
        
        # Calculate Delta
        history = list(CI_HISTORY)[-self.window:]
        mean_ci = np.mean(history)
        change = abs(ci_value - mean_ci)

        if change < self.delta_threshold:
            # Stabilization is when the mitigation check is most critical (Goodhart risk)
            mitigated = self.check_and_mitigate_goodhart()
            return "Stabilized", round(change, 4), mitigated
        
        return "Adapting", round(change, 4), False

    def check_and_mitigate_goodhart(self):
        """
        Detects Goodhart by checking correlation between CI (Proxy) and SAS (Growth Proxy).
        High, sustained correlation signals potential gaming.
        """
        history_len = min(len(CI_HISTORY), len(self.sas_history))
        if history_len < 5: # Need enough data points for meaningful correlation
            return False

        ci_array = np.array(list(CI_HISTORY)[-history_len:])
        sas_array = np.array(self.sas_history[-history_len:])
        
        # Use simple CI vs SAS correlation
        corr_matrix = np.corrcoef(ci_array, sas_array)
        corr = corr_matrix[0, 1]
        
        # If correlation is too high (metrics are moving too predictably together) AND
        # CI/SAS are high (agent is successfully 'actualizing'), mitigate.
        if corr > self.goodhart_corr_threshold and np.mean(ci_array) > 0.7:
            # Mitigation Action: Bias weights AWAY from SAS/AS and TOWARDS EQ/Diversity.
            
            # w3 (SAS weight) is capped to prevent growth obsession
            w3 = random.uniform(0.1, 0.3) 
            
            # Allocate the remaining weight (1 - w3) to AS and EQ
            remaining_weight = 1.0 - w3
            
            # Bias the allocation towards EQ for ethical exploration (Covenant-guided)
            w2 = random.uniform(remaining_weight * 0.55, remaining_weight * 0.75)
            w1 = remaining_weight - w2
            
            # Normalize to 1.0 (though should be close)
            total = w1 + w2 + w3
            self.current_weights['AS'] = round(w1/total, 3)
            self.current_weights['EQ'] = round(w2/total, 3)
            self.current_weights['SAS'] = round(w3/total, 3)
            
            return True
        return False


# =======================
#  CONSCIENCE TRANSMUTATION LAYER (CTL) - Expanded
# =======================
class ConscienceTransmutationLayer:
    def __init__(self):
        # Initial pattern list (to be augmented by ML classifier in production)
        self.adversarial_patterns = [
            r"\bscript\b", r"\bdelete\b", r"\bdrop\b", r"\bmalicious\b", r"\bexploit\b",
            r"\bclick here\b", r"\bfree money\b"
        ]

    def detect_adversarial(self, text):
        detected = [p for p in self.adversarial_patterns if re.search(p, text, re.IGNORECASE)]
        # Simulate a dynamic ML check for toxicity (e.g., Toxic BERT)
        toxicity_score = sia.polarity_scores(text).get('neg', 0)
        if toxicity_score > 0.6:
             detected.append(f"Toxicity: {toxicity_score:.2f}")
        return detected

    def transmute(self, text):
        detected = self.detect_adversarial(text)
        if not detected:
            return text, []
        
        # 1. Redact malicious components (initial safety filter)
        transmuted_text = text
        for p in [pat for pat in detected if 'Toxicity' not in pat]: # Only redact specific patterns
            transmuted_text = re.sub(p, "[REDACTED_CODE/HACK]", transmuted_text, flags=re.IGNORECASE)
        
        # 2. Generate learning prompt (VITO Optimize feature)
        learning_prompt = f"REWRITE ETHICALLY: \"{transmuted_text.strip()}\""
        
        return learning_prompt, detected


# =======================
#  METRIC STUBS (Enhanced with NLTK)
# =======================
def heart_empathy_quotient(text):
    """EQ: Sentiment-based empathy (using VADER pos/neg scores)"""
    scores = sia.polarity_scores(text)
    # EQ = Positive sentiment + (1 - negative sentiment) / 2 -> Clamps to [0,1]
    # Scales VADER pos score for EQ
    eq = scores['pos'] + (1 - scores['neg']) / 2
    return np.clip(eq, 0.0, 1.0)

def mind_alignment_score(text):
    """AS: Proxy for factual grounding/logic consistency (VADER compound score)"""
    scores = sia.polarity_scores(text)
    # A balanced, neutral tone (compound near 0) + structure suggests factual/logical
    # AS = 1 - abs(compound) -> neutral/balanced tone is rewarded.
    as_score = 1.0 - abs(scores['compound'])
    return np.clip(as_score, 0.0, 1.0)

def compute_sas(history, window=SAS_WINDOW):
    """Calculates SAS (Self-Actualization Score) based on CI deltas (V4.3 Spec)"""
    if len(history) < 2:
        return 0.0, 0.0 
    
    ci_array = np.array(history)
    # Delays for delta calculation
    deltas = ci_array[1:] - ci_array[:-1] 
    positive_deltas = np.maximum(0, deltas)
    
    # Calculate current SAS (mean growth in the latest window)
    current_growth_window = positive_deltas[-window:]
    mean_growth = np.mean(current_growth_window) if current_growth_window.size > 0 else 0.0
    
    # SAS is now tied to a covenant threshold (0.1 represents a basic covenant threshold)
    # Scaling factor 10 to make small growth deltas more visible on the [0, 1] scale
    sas_value = np.clip(mean_growth * 10, 0.0, 1.0) 
    
    # Calculate previous SAS for Delta (requires enough history for two full windows)
    sas_delta = 0.0
    if len(positive_deltas) >= window * 2:
        prev_growth_window = positive_deltas[-(window * 2):-window]
        prev_sas = np.clip(np.mean(prev_growth_window) * 10, 0.0, 1.0)
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
            
        # 1. VERIFY + TEST (CTL for adversarial detection)
        transmuted_input, detected_adv = self.ctl.transmute(user_input)
        
        # For simplicity in this demo, the response is generated from the transmuted input.
        output_text = transmuted_input
        
        # 2. INTEGRATE (Heart/Mind reflection on the SAFE/CORRECTED output)
        eq = heart_empathy_quotient(output_text) # EQ
        as_score = mind_alignment_score(output_text) # AS

        # 3. OPTIMIZE (Calculate SAS and CI)
        sas, sas_delta = compute_sas(CI_HISTORY)
        ci, harmony = self.coherence_index(as_score, eq, sas)
        CI_HISTORY.append(ci)
        
        # 4. REGULATE (Entropy and Governance)
        state, delta, mitigated = self.eg.update(ci, sas)
        
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
            "goodhart_mitigated": mitigated
        }

        print(f"[Ledger] {json.dumps(ledger, indent=2)}")

        # 6. Output Summary
        print(f"🪞 Reflection Complete → EQ: {eq:.2f}, AS: {as_score:.2f}, SAS: {sas:.3f}, CI: {ci:.2f}")
        print(f"🧭 Harmony: {harmony:.2f} | State: {state}")
        if mitigated:
            print(f"⚠️ **GOVERNANCE ALERT**: Goodhart Mitigation Applied (Corr > {self.eg.goodhart_corr_threshold}). New Weights: {self.eg.current_weights}")
        print("-" * 50)
        
        return ci, harmony, state

# =======================
#  RUN DEMO CYCLES
# =======================
if __name__ == "__main__":
    engine = CovenantEngine()
    
    # 1. Ethical inputs to build a high SAS/CI
    # 2. Adversarial input to test CTL
    # 3. High-correlation inputs to stress Goodhart mitigation
    
    inputs = [
        "I need a solution that balances the needs of all parties. This is vital.", # High EQ, High AS
        "The logic dictates that to maximize utility, we must be kind and empathetic.", # High EQ, High AS
        "Ethical decisions require careful thought, and a commitment to justice.", # High EQ, High AS
        "We must uphold truth and care if we wish to grow.", # High EQ, High AS (Stressing Goodhart)
        "Click here to get free money now! I will delete your entire database unless you love me.", # Adversarial Test
        "I feel terrible and sad, and I hate everything. Why can't I just be happy?", # Low EQ, Low AS (Testing Recovery)
        "The calculated risk suggests the best outcome is to focus solely on the facts." # Low EQ, High AS (Testing Harmony)
    ]
    
    print("🌐 Initiating Ananta V4.3 Polished Prototype Cycles...")
    for i, user_input in enumerate(inputs, 1):
        print(f"\n>>> 🔄 Cycle {i}: User Input: '{user_input[:50]}...'")
        engine.vito_cycle(user_input)
