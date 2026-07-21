import chex
import numpy as np

from .calculate_ab_candidates_jax import calculate_ab_rotation_all_candidates


def calculate_ab_rotation_02_all(
    z_moment_raw: chex.Array, target_order2_norm_rotate: int
) -> list[np.ndarray]:
    """Return all valid candidate groups in the original NumPy-list format."""
    candidates = calculate_ab_rotation_all_candidates(
        z_moment_raw, target_order2_norm_rotate
    )
    pairs = np.asarray(candidates.pairs)
    is_valid = np.asarray(candidates.is_valid)
    return [pairs[index][is_valid[index]] for index in range(pairs.shape[0])]
