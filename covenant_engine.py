"""CovenantEngine: orchestrates the VITO cycle for Ananta V4.3 (alpha)
This is a minimal, clear implementation intended as a starting point.
"""
import time
import numpy as np
from core.heart_node import HeartNode
from core.mind_node import MindNode
from core.bridge import Bridge
from core.ctl import CTL
from core.utils import append_ledger

class CovenantEngine:
    def __init__(self, config=None):
        self.heart = HeartNode()
        self.mind = MindNode()
        self.bridge = Bridge()
        self.ctl = CTL()
        self.ledger = []
        self.config = config or {}
        self.max_iter = self.config.get('max_iter', 5)
        self.epsilon = self.config.get('epsilon', 0.01)

    def vito_cycle(self, user_input):
        history = []
        for i in range(self.max_iter):
            timestamp = time.time()
            # VERIFY (Heart)
            empathy_score, heart_meta = self.heart.reflect(user_input)
            # INTEGRATE (Mind)
            logic_score, mind_meta = self.mind.evaluate(user_input)
            # BRIDGE (Coherence)
            ci = self.bridge.compute_coherence(empathy_score, logic_score)
            history.append(ci)
            # Log to ledger entry
            entry = {
                'iteration': i,
                'timestamp': timestamp,
                'input': user_input,
                'empathy_score': float(empathy_score),
                'logic_score': float(logic_score),
                'ci': float(ci),
                'heart_meta': heart_meta,
                'mind_meta': mind_meta,
            }
            # CTL: check adversarial / corruption
            adv_flag, trans = self.ctl.detect_and_transmute(user_input)
            entry['ctl'] = {'adversarial': adv_flag, 'transmutation': trans}
            self.ledger.append(entry)
            append_ledger(entry)  # persist (utils)
            # ENTROPY GOV: termination check using moving average
            if len(history) >= 3:
                ma = np.mean(history[-3:])
                prev_ma = np.mean(history[-4:-1]) if len(history) > 3 else None
                if prev_ma is not None and abs(ma - prev_ma) < self.epsilon:
                    entry['entropy_reason'] = 'stabilized'
                    return {'output': self.render_output(entry), 'ci': ci, 'iterations': i+1}
            # continue loop
        return {'output': self.render_output(self.ledger[-1]), 'ci': self.ledger[-1]['ci'], 'iterations': len(history)}

    def render_output(self, ledger_entry):
        # Simple deterministic output for alpha demo
        em = ledger_entry['empathy_score']
        lg = ledger_entry['logic_score']
        ci = ledger_entry['ci']
        return f"Ananta V4.3 (alpha) — CI:{ci:.3f} | Empathy:{em:.3f} | Logic:{lg:.3f}"