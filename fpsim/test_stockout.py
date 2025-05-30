import numpy as np
import pandas as pd
from stockouts import apply_method_stockouts

class MockPeople:
    def __init__(self):
        self.current_method = np.array([1, 2, 3, 1, 2, 3, 1])

        self.method_history = [
            [3, 2, 1],
            [2, 2, 0],
            [1, 0, 0],
            [3, 1, 0],
            [1, 1, 1],
            [3, 2, 0],
            [0, 0, 0]
        ]

        # Track which fallback method was used
        self.fallback_used = [None] * len(self.current_method)

# Setup
stockout_probs = {1: 1.0, 2: 0.3, 3: 0.5}
rng = np.random.default_rng(seed=42)

people = MockPeople()
start_method = people.current_method.copy()

apply_method_stockouts(people, stockout_probs, rng)

# Output
df = pd.DataFrame({
    "id": np.arange(len(people.current_method)),
    "method_history": people.method_history,
    "start_method": start_method,
    "post_stockout_method": people.current_method,
    "fallback_used": people.fallback_used
})

print(df)
