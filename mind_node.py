"""Mind Node: logic/factuality wrapper (alpha)
Uses a classifier or heuristic to produce a logic_score (0-1) and metadata.
"""
import random
import numpy as np
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

class MindNode:
    def __init__(self, model_name='roberta-base'):
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            except Exception:
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None

    def evaluate(self, text):
        """Return logic_score (0-1) and metadata dict."""
        if self.model is None:
            # simple heuristic: more punctuation -> assumed complexity -> higher logic score
            score = min(0.95, max(0.05, (text.count('.') + text.count('?')) / 10.0))