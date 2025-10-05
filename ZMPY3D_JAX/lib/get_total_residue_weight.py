# This function primarily involves dictionary lookups and a loop.
# While JAX can handle loops, it's generally more efficient to vectorize operations.
# If `aa_name_list` and `residue_weight_map` can be represented as JAX arrays (e.g., one-hot encoded amino acid names and a weight array), the summation could be vectorized using `jax.numpy.sum` and `jax.numpy.take` or similar operations for efficient JAX compilation.
# The dictionary lookup for missing amino acids would need careful handling in a JAX-compatible way (e.g., using `jax.lax.select` with a boolean mask).

from typing import Dict, Sequence


def get_total_residue_weight(
    aa_name_list: Sequence[str], residue_weight_map: Dict[str, float]
) -> float:
    """Calculates the total residue weight of a protein given a list of amino acid names
    and a mapping of amino acid names to their weights.
    Defaults to 'ASP' if an amino acid name is not found in the map.

    Args:
        aa_name_list (list): A list of three-letter amino acid codes.
        residue_weight_map (dict): A dictionary mapping three-letter amino acid codes to their weights.

    Returns:
        float: The total residue weight of the protein.
    """
    weight_multiplier = 1

    total_residue_weight = 0
    for aa_name in aa_name_list:
        if aa_name not in residue_weight_map:
            aa_name = "ASP"
        total_residue_weight += residue_weight_map[aa_name] * weight_multiplier

    return total_residue_weight
