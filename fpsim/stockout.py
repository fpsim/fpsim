import numpy as np

def apply_method_stockouts(people, stockout_probs, rng=None):
    """
    If current method stockouts, try fallback methods in method_history.
    Track fallback used in people.fallback_used if present.
    """
    if rng is None:
        rng = np.random.default_rng()

    users = np.where(people.current_method > 0)[0]

    for i in users:
        current = people.current_method[i]
        prob = stockout_probs.get(current, 0.0)

        if rng.random() < prob:
            fallback = 0
            fallback_source = 'none'

            for past_method in people.method_history[i]:
                if past_method > 0:
                    fallback_prob = stockout_probs.get(past_method, 0.0)
                    if rng.random() >= fallback_prob:
                        fallback = past_method
                        fallback_source = past_method
                        break

            people.current_method[i] = fallback

            # This is safe: only called if people object has the attribute
            if hasattr(people, "fallback_used"):
                people.fallback_used[i] = fallback_source
