"""Bridge: compute Coherence Index (CI) from empathy and logic scores."""
import numpy as np

class Bridge:
    def __init__(self, w1=0.4, w2=0.4, w3=0.2):
        self.w1 = w1  # alignment / intent weight (placeholder)
        self.w2 = w2  # empathy weight
        self.w3 = w3  # SAS weight (uses small historical proxy here)

    def compute_coherence(self, empathy_score, logic_score, sas=0.5):
        # clamp inputs
        e = float(np.clip(empathy_score, 0.0, 1.0))
        l = float(np.clip(logic_score, 0.0, 1.0))
        sas = float(np.clip(sas, 0.0, 1.0))
        ci = self.w1 * e + self.w2 * l + self.w3 * sas
        return float(np.clip(ci, 0.0, 1.0))