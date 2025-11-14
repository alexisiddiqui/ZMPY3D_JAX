"""
Benchmark: Processing list of 3D arrays with different sizes

Compares two approaches:
1. Direct: Loop through arrays individually, process each
2. Padded + vmap: Pad all to max size, stack, use vmap for parallel processing
"""

import time
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ZMPY3D_JAX as z


def generate_heterogeneous_arrays(num_arrays: int, min_size: int = 5, max_size: int = 20):
    """Generate a list of 3D arrays with varying sizes."""
    arrays = []
    sizes = []

    for i in range(num_arrays):
        # Random size for each array
        size_x = np.random.randint(min_size, max_size + 1)
        size_y = np.random.randint(min_size, max_size + 1)
        size_z = np.random.randint(min_size, max_size + 1)

        # Create array with some density
        voxel = np.zeros((size_x, size_y, size_z))
        # Add random sparse density
        density_points = np.random.randint(5, 20)
        for _ in range(density_points):
            x = np.random.randint(0, size_x)
            y = np.random.randint(0, size_y)
            z = np.random.randint(0, size_z)
            voxel[x, y, z] = np.random.random()

        arrays.append(jnp.asarray(voxel))
        sizes.append((size_x, size_y, size_z))

    return arrays, sizes


def create_xyz_samples(shape):
    """Create xyz_sample_struct for a given 3D shape."""
    return {
        "X_sample": jnp.arange(shape[0] + 1, dtype=jnp.float32),
        "Y_sample": jnp.arange(shape[1] + 1, dtype=jnp.float32),
        "Z_sample": jnp.arange(shape[2] + 1, dtype=jnp.float32),
    }


# ============================================================================
# Approach 1: Direct Processing (Loop)
# ============================================================================

def process_heterogeneous_direct(arrays, max_order=1):
    """Process each array individually in a loop."""
    results = []

    for voxel in arrays:
        xyz_samples = create_xyz_samples(voxel.shape)
        volume_mass, center, moment = z.calculate_bbox_moment(voxel, max_order, xyz_samples)
        results.append({
            "volume_mass": volume_mass,
            "center": center,
            "moment": moment,
        })

    return results


# ============================================================================
# Approach 2: Padded + vmap
# ============================================================================

def pad_arrays_to_max(arrays):
    """Pad all arrays to maximum size."""
    max_x = max(arr.shape[0] for arr in arrays)
    max_y = max(arr.shape[1] for arr in arrays)
    max_z = max(arr.shape[2] for arr in arrays)

    padded_arrays = []
    for arr in arrays:
        pad_x = max_x - arr.shape[0]
        pad_y = max_y - arr.shape[1]
        pad_z = max_z - arr.shape[2]

        padded = jnp.pad(arr, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant', constant_values=0)
        padded_arrays.append(padded)

    return jnp.stack(padded_arrays), (max_x, max_y, max_z)


def process_single_array_for_vmap(voxel, xyz_samples_dict):
    """Process a single array - designed for vmap."""
    volume_mass, center, moment = z.calculate_bbox_moment(voxel, 1, xyz_samples_dict)
    return volume_mass, center, moment


def process_heterogeneous_vmap(arrays, max_order=1):
    """Process arrays using vmap after padding."""
    stacked_arrays, max_shape = pad_arrays_to_max(arrays)

    # Create xyz_samples for the maximum shape
    xyz_samples = create_xyz_samples(max_shape)

    # Apply vmap over the batch dimension (first dimension of stacked_arrays)
    # vmap will map over axis 0 of the voxel array
    vmap_func = jax.vmap(lambda voxel: process_single_array_for_vmap(voxel, xyz_samples))

    volume_masses, centers, moments = vmap_func(stacked_arrays)

    return [
        {
            "volume_mass": volume_masses[i],
            "center": centers[i],
            "moment": moments[i],
        }
        for i in range(len(arrays))
    ]


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark():
    """Run comprehensive benchmark."""
    print("=" * 80)
    print("HETEROGENEOUS 3D ARRAY PROCESSING BENCHMARK")
    print("=" * 80)

    test_configs = [
        (5, "Small batch"),
        (10, "Medium batch"),
        (20, "Large batch"),
    ]

    for num_arrays, label in test_configs:
        print(f"\n{'=' * 80}")
        print(f"Test: {label} ({num_arrays} arrays)")
        print(f"{'=' * 80}")

        # Generate test data
        arrays, sizes = generate_heterogeneous_arrays(num_arrays, min_size=5, max_size=20)

        # Print array info
        print(f"\nArray sizes:")
        for i, (x, y, z) in enumerate(sizes):
            print(f"  Array {i}: {x}×{y}×{z}")

        min_size = min(np.prod(s) for s in sizes)
        max_size = max(np.prod(s) for s in sizes)
        avg_size = np.mean([np.prod(s) for s in sizes])
        print(f"\nSize statistics:")
        print(f"  Min: {min_size:,} elements")
        print(f"  Max: {max_size:,} elements")
        print(f"  Avg: {avg_size:,.0f} elements")
        print(f"  Max total (stacked): {max_size * num_arrays:,} elements")

        # Approach 1: Direct processing
        print(f"\n{'─' * 80}")
        print("APPROACH 1: Direct Processing (Loop)")
        print(f"{'─' * 80}")

        # Warm-up
        _ = process_heterogeneous_direct(arrays)

        repeats = 10
        start = time.perf_counter()
        for _ in range(repeats):
            results_direct = process_heterogeneous_direct(arrays)
        elapsed_direct = time.perf_counter() - start

        time_per_iter_direct = elapsed_direct / repeats * 1000  # ms

        print(f"Repeats: {repeats}")
        print(f"Total time: {elapsed_direct:.4f}s")
        print(f"Time per iteration: {time_per_iter_direct:.4f}ms")
        print(f"Arrays per second: {repeats / elapsed_direct:.2f}")

        # Memory estimate
        total_elements_direct = sum(np.prod(arr.shape) for arr in arrays)
        print(f"Memory (direct): {total_elements_direct * 4 / 1e6:.2f}MB (float32)")

        # Approach 2: Padded + vmap
        print(f"\n{'─' * 80}")
        print("APPROACH 2: Padded + vmap")
        print(f"{'─' * 80}")

        # Pre-compute padding info
        max_x = max(arr.shape[0] for arr in arrays)
        max_y = max(arr.shape[1] for arr in arrays)
        max_z = max(arr.shape[2] for arr in arrays)
        padded_shape = (max_x, max_y, max_z)

        print(f"Padded shape: {padded_shape}")
        padded_elements_per_array = np.prod(padded_shape)
        padding_overhead = (padded_elements_per_array * num_arrays - total_elements_direct) / total_elements_direct * 100
        print(f"Elements per padded array: {padded_elements_per_array:,}")
        print(f"Memory (padded): {padded_elements_per_array * num_arrays * 4 / 1e6:.2f}MB (float32)")
        print(f"Padding overhead: {padding_overhead:.1f}%")

        # Warm-up
        _ = process_heterogeneous_vmap(arrays)

        start = time.perf_counter()
        for _ in range(repeats):
            results_vmap = process_heterogeneous_vmap(arrays)
        elapsed_vmap = time.perf_counter() - start

        time_per_iter_vmap = elapsed_vmap / repeats * 1000  # ms

        print(f"\nRepeats: {repeats}")
        print(f"Total time: {elapsed_vmap:.4f}s")
        print(f"Time per iteration: {time_per_iter_vmap:.4f}ms")
        print(f"Arrays per second: {repeats / elapsed_vmap:.2f}")

        # Comparison
        print(f"\n{'─' * 80}")
        print("COMPARISON")
        print(f"{'─' * 80}")

        speedup = elapsed_direct / elapsed_vmap
        time_diff = time_per_iter_vmap - time_per_iter_direct

        if speedup > 1:
            print(f"✓ vmap is {speedup:.2f}x FASTER")
        else:
            print(f"✗ Direct is {1/speedup:.2f}x FASTER")

        print(f"Time difference: {time_diff:+.4f}ms per iteration")

        # Verify results match
        volumes_direct = [r["volume_mass"] for r in results_direct]
        volumes_vmap = [r["volume_mass"] for r in results_vmap]
        volumes_match = np.allclose(volumes_direct, volumes_vmap, rtol=1e-5)
        print(f"Results match: {'✓ YES' if volumes_match else '✗ NO'}")


if __name__ == "__main__":
    benchmark()
