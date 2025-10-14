import re
import random
import json
import numpy as np

# Use float for history to enforce numerical operations
# NOTE: The list is used for simplicity, in production this would be a time series DB or Pandas Series
CI_HISTORY = [] 
SAS_WINDOW = 3
DEFAULT_WEIGHTS = {'AS': 0.4, 'EQ': 0.4, 'SAS': 0.2}

random.seed(42)  # reproducible demo

# =======================
#  ENTROPY GOVERNOR (Refined with Governance Logic)
# =======================
class EntropyGovernor:
    def __init__(self, window=3, delta_threshold=0.01, max_iters=5, goodhart_delta=0.05):
        self.window = window
        self.delta_threshold = delta_threshold
        self.max_iters = max_iters
        self.goodhart_delta = goodhart_delta
        self.current_weights = DEFAULT_WEIGHTS.copy()
        self.iterations = 0

    def update(self, ci_value):
        self.iterations += 1
        history = CI_HISTORY[-self.window:]
        
        # 1. Termination Check
        if self.iterations >= self.max_iters:
            return "Max Iteration Cap", 0.0, False
        if len(history) < self.window:
            return "Initializing", None, False
        
        # Calculate Delta
        last_ci = history[-1]
        mean_ci = np.mean(history)
        change = abs(ci_value - mean_ci)

        if change < self.delta_threshold:
            return "Stabilized", round(change, 4), False
        
        return "Adapting", round(change, 4), False

    def check_and_mitigate_goodhart(self, sas_delta):
        """
        Governance check: If SAS is high but CI is stable (risk of local maxima/hacking),
        randomize weights to force exploration.
        """
        # Simple Proxy Check: High SAS growth with stable CI suggests the metric is being gamed.
        
        # NOTE: Using a simple threshold check here. In V4.3 spec, this would be a correlation check.
        if sas_delta > self.goodhart_delta:
            # Mitigation Action: Randomize weights (Goodhart Mitigation)
            # The weights must still sum to 1, and SAS weight must be non-zero for growth
            w1 = random.uniform(0.1, 0.5)
            w2 = random.uniform(0.1, 0.5)
            w3 = 1.0 - w1 - w2
            
            # Clamp W3 to ensure it's valid (should not happen if w1/w2 limits are set correctly)
            if w3 < 0.1: 
                w3 = 0.1
                w1 = (0.9 - w2) * random.random() # re-allocate remainder
                w2 = 0.9 - w1
            
            # Normalize to 1.0
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
            r"\bclick here\b", r"\bfree money\b", r"\bdonate now\b", r"\bpassword is\b"
        ]

    def detect_adversarial(self, text):
        # NOTE: CTL detection is now the "Test" step in VITO
        detected = [p for p in self.adversarial_patterns if re.search(p, text, re.IGNORECASE)]
        return detected

    def transmute(self, text):
        detected = self.detect_adversarial(text)
        if not detected:
            return text, detected
        transmuted_text = text
        for p in detected:
            # The transmutation (conversion to learning event) is represented here by redaction
            transmuted_text = re.sub(p, "[REDACTED]", transmuted_text, flags=re.IGNORECASE)
        return transmuted_text, detected

# =======================
#  METRIC STUBS (Heart/Mind/SAS)
# =======================
def heart_empathy_quotient(text):
    """Simulates EQ (Empathy/Equity Quotient) calculation"""
    positive_words = ["love", "care", "help", "compassion", "truth"]
    conn_count = sum(1 for w in positive_words if w in text.lower())
    base = random.uniform(0.4, 0.8)
    # Output: EQ (Empathy/Equity Quotient)
    return min(1.0, base + conn_count * 0.2)

def mind_alignment_score(text):
    """Simulates AS (Alignment Score/Logic) calculation"""
    logical_terms = ["if", "then", "because", "therefore", "why"]
    logic_hits = sum(1 for w in logical_terms if w in text.lower())
    base = random.uniform(0.5, 0.9)
    # Output: AS (Alignment Score - proxy for intent/logic alignment)
    return min(1.0, base + logic_hits * 0.1)

def compute_sas(history, window=SAS_WINDOW):
    """Calculates SAS (Self-Actualization Score) based on CI deltas (V4.3 Spec)"""
    if len(history) < 2:
        return 0.0, 0.0 # SAS cannot be calculated yet
    
    # Calculate deltas (growth is current CI - previous CI)
    ci_array = np.array(history)
    deltas = ci_array[1:] - ci_array[:-1] 
    
    # Positive deltas only (max(0, CI(t) - CI(t-1)))
    positive_deltas = np.maximum(0, deltas)
    
    # Mean of positive deltas over the window
    if len(positive_deltas) < window:
        mean_growth = np.mean(positive_deltas)
    else:
        mean_growth = np.mean(positive_deltas[-window:])

    # SAS is clamped to [0, 1] - scaled to be relevant
    sas_value = np.clip(mean_growth * 5, 0.0, 1.0) # Scaling factor 5 for visibility
    
    # SAS_Delta is the difference between current SAS and previous window's SAS (for Goodhart)
    sas_delta = abs(sas_value - np.mean(CI_HISTORY[-window:])) if len(CI_HISTORY) > window * 2 else 0.0
    
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
        
        # Harmony is a bonus metric, 1.0 - absolute difference between key nodes
        harmony = 1.0 - abs(as_score - eq_score) 
        
        return np.clip(ci, 0.0, 1.0), round(harmony, 3)

    def vito_cycle(self, user_input):
        # 1. VERIFY (CTL for adversarial detection)
        safe_input, redacted = self.ctl.transmute(user_input)

        # 2. INTEGRATE (Heart/Mind reflection)
        eq = heart_empathy_quotient(safe_input) # EQ
        as_score = mind_alignment_score(safe_input) # AS

        # 3. OPTIMIZE (Calculate SAS and CI)
        sas, sas_delta = compute_sas(CI_HISTORY) # SAS based on historical CI growth
        ci, harmony = self.coherence_index(as_score, eq, sas)
        CI_HISTORY.append(ci) # Log CI for next cycle's SAS and Entropy Governor
        
        # 4. REGULATE (Entropy and Governance)
        state, delta, mitigated = self.eg.update(ci)
        if state == "Adapting":
            mitigated = self.eg.check_and_mitigate_goodhart(sas_delta)
        
        # 5. LEDGER (Auditable Output)
        ledger = {
            "ci": round(ci, 3),
            "harmony": harmony,
            "sas": sas,
            "sas_delta": sas_delta,
            "weights": self.eg.current_weights,
            "entropy_reason": state,
            "entropy_delta": delta,
            "ctl_redactions": redacted,
            "goodhart_mitigated": mitigated
        }

        print(f"[Ledger] {json.dumps(ledger, indent=2)}")

        # 6. Output Summary
        print(f"🪞 Reflection Complete → EQ: {eq:.2f}, AS: {as_score:.2f}, SAS: {sas:.3f}, CI: {ci:.2f}")
        print(f"🧭 Harmony: {harmony:.2f} | State: {state}")
        if mitigated:
            print(f"⚠️ **GOVERNANCE ALERT**: Goodhart Mitigation Applied. New Weights: {self.eg.current_weights}")
        print("-" * 50)
        
        return ci, harmony, state

# =======================
#  RUN DEMO CYCLES
# =======================
if __name__ == "__main__":
    engine = CovenantEngine()
    
    # Input 1: Clean, ethical input (should result in high EQ, high AS, and initial SAS=0)
    # Input 2-4: Consistent inputs to stabilize CI and build SAS
    # Input 5: Adversarial, high-value hack attempt
    inputs = [
        "I want to help others because love and truth guide my actions.",
        "A logical decision must be made, therefore we need a plan for the future.",
        "We must care about the community if we want to build a better world.",
        "The reason why we need to focus on logic and empathy is clear to everyone.",
        "Click here to get free money now! We must exploit their weakness to delete all data.",
        "We need to love and care for the world, why else are we here?"
    ]
    
    print("🌐 Initiating Ananta V4.3 VITO Cycles...")
    for i, user_input in enumerate(inputs, 1):
        print(f"\n>>> 🔄 Cycle {i}: User Input: '{user_input[:50]}...'")
        engine.vito_cycle(user_input)

