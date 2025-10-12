"""Heart Node: empathy embedding wrapper (alpha)
Lightweight: uses transformers to produce an 'empathy score' scalar and metadata.
"""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class HeartNode:
    def __init__(self, model_name='bert-base-uncased'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            # fallback: dummy embeddings
            self.tokenizer = None
            self.model = None

    def reflect(self, text):
        """Return empathy_score (0-1) and metadata dict."""
        if self.model is None:
            # deterministic pseudo-score for offline/demo use
            score = (len(text) % 10) / 10.0
            meta = {'mode': 'dummy'}
            return score, meta
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state.mean(dim=1)
        vec = outputs.squeeze().mean().item()
        # normalize
        score = 1.0 / (1.0 + np.exp(-vec/100.0))
        meta = {'mode': 'model', 'vec_mean': float(vec)}
        return float(score), meta