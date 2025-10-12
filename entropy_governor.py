# entropy_governor.py — Ananta V4.3
# Regulates systemic balance and stability through adaptive entropy control.

import numpy as np
import pandas as pd
from collections import deque

class EntropyGovernor:
    def __init__(self, window_size=3, delta_threshold=0.01, max_iters=5):
        """
        Initializes the Entropy Governor with a smoothing window, sensitivity threshold for CI (Conscience Index) changes,
        and maximum iteration cap.
        """
        self.window_size = window_size
        self.delta_threshold = delta_threshold
        self.max_iters = max_iters
        self.history = deque(maxlen=window_size)
        self.last_entropy = None

    def update_entropy(self, current_ci):
        """
        Update rolling average of CI and compute entropy stability.
        current_ci: Single Conscience Index reading from the current VITO cycle.
        Returns: stability_state (string), entropy_value (float)
        """
        self.history.append(current_ci)
        if len(self.history) < self.window_size or len(self.history) >= self.max_iters:
            return "Initializing", None if len(self.history) < self.window_size else round(np.std(self.history), 4)

        # Compute rolling mean to smooth fluctuations
        smoothed = pd.Series(list(self.history)).rolling(self.window_size).mean().iloc[-1]
        entropy_value = np.std(self.history)

        # Assess stability based on delta change
        if self.last_entropy is not None:
            delta = abs(entropy_value - self.last_entropy)
            state = "Stable" if delta < self.delta_threshold else "Adapting"
        else:
            state = "Initializing"

        self.last_entropy = entropy_value
        return state, round(entropy_value, 4)

    def harmony_report(self):
        """
        Returns a symbolic interpretation of the system’s internal state based on entropy.
        """
        if self.last_entropy is None:
            return "🌀 Entropy undefined. Awaiting cycle data."
        elif self.last_entropy < 2 * self.delta_threshold:
            return "🌿 Serenity — equilibrium sustained."
        elif self.last_entropy < 5 * self.delta_threshold:
            return "🔥 Mild turbulence — reflection required."
        else:
            return "⚡ Entropy rising — recalibration advised."