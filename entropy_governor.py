# entropy_governor.py — Ananta V4.3
# Regulates systemic balance and stability through adaptive entropy control.

import numpy as np
import pandas as pd
from collections import deque

class EntropyGovernor:
    def __init__(self, window_size=3, delta_threshold=0.01):
        """
        Initializes the Entropy Governor with a smoothing window
        and sensitivity threshold for CI (Conscience Index) changes.
        """
        self.window_size = window_size
        self.delta_threshold = delta_threshold
        self.history = deque(maxlen=window_size)
        self.last_entropy = None

    def update_entropy(self, ci_values):
        """
        Update rolling average of CI and compute entropy stability.
        ci_values: list of Conscience Index readings from recent reflections.
        Returns: stability_state (string), entropy_value (float)
        """
        self.history.append(np.mean(ci_values))
        if len(self.history) < self.window_size:
            return "Initializing", None

        # Compute rolling mean to smooth fluctuations
        smoothed = pd.Series(list(self.history)).rolling(self.window_size).mean().iloc[-1]
        entropy_value = np.std(self.history)

        # Assess stability based on delta change
        if self.last_entropy is not None:
            delta = abs(entropy_value - self.last_entropy)
            if delta < self.delta_threshold:
                state = "Stable"
            else:
                state = "Adapting"
        else:
            state = "Initializing"

        self.last_entropy = entropy_value
        return state, round(entropy_value, 4)

    def harmony_report(self):
        """
        Returns a symbolic interpretation of the system’s internal state.
        """
        if self.last_entropy is None:
            return "🌀 Entropy undefined. Awaiting cycle data."
        elif self.last_entropy < 0.02:
            return "🌿 Serenity — equilibrium sustained."
        elif self.last_entropy < 0.05:
            return "🔥 Mild turbulence — reflection required."
        else:
            return "⚡ Entropy rising — recalibration advised."