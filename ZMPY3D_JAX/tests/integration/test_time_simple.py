import os
import time
import pickle
import pytest
import logging
from datetime import datetime

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default

def test_time_evaluation_runtime():
    """
    Integration-like runtime test that measures the time to run the core Zernike conversion
    pipeline repeatedly, following the notebook evaluation methodology exactly.
    Uses NumPy arrays as in the notebook's final evaluation section.
    Skips if ZMPY3D_JAX or its cache is unavailable.
    Configurable via environment variables:
      - ZMPY3D_TIME_REPEATS (default 10000, matching notebook)
      - ZMPY3D_TIME_MAX_SEC (default 1200 = 20 minutes)
      - ZMPY3D_TIME_MAXORDER (default 20)
    """
    # Config - defaults match notebook behavior
    repeats = _env_int("ZMPY3D_TIME_REPEATS", 100)
    max_seconds = _env_int("ZMPY3D_TIME_MAX_SEC", 1200)  # 20 minutes default
    max_order = _env_int("ZMPY3D_TIME_MAXORDER", 20)

    try:
        import ZMPY3D_JAX as z
        import numpy as np
    except Exception as e:
        pytest.skip(f"ZMPY3D_JAX or numpy not available: {e}")

    # Load global parameters
    Param = z.get_global_parameter()

    # Try to load precomputed cache; skip test if not found
    cache_path = None
    try:
        # Find cache_data directory based on the site package location of ZMPY3D
        cache_dir = z.__file__.replace("__init__.py", "cache_data")
        cache_path = os.path.join(cache_dir, f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl")
        with open(cache_path, "rb") as file:
            CachePKL = pickle.load(file)
    except Exception as e:
        pytest.skip(f"Precomputed cache not available at {cache_path}: {e}")

    # Extract cache data exactly as in notebook
    GCache_pqr_linear = CachePKL['GCache_pqr_linear']
    GCache_complex = CachePKL['GCache_complex']
    GCache_complex_index = CachePKL['GCache_complex_index']
    CLMCache3D = CachePKL['CLMCache3D']
    CLMCache = CachePKL['CLMCache']

    # Define OneTimeConversion following the notebook exactly
    def OneTimeConversion(Voxel3D, MaxOrder):
        """Follows the notebook's NumPy implementation exactly"""
        Dimension_BBox_scaled = Voxel3D.shape
        
        XYZ_SampleStruct = {
            'X_sample': np.arange(Dimension_BBox_scaled[0] + 1),
            'Y_sample': np.arange(Dimension_BBox_scaled[1] + 1),
            'Z_sample': np.arange(Dimension_BBox_scaled[2] + 1)
        }
        
        [VolumeMass, Center, _] = z.calculate_bbox_moment(Voxel3D, 1, XYZ_SampleStruct)
        
        [AverageVoxelDist2Center, _] = z.calculate_molecular_radius(
            Voxel3D, Center, VolumeMass, Param['default_radius_multiplier']
        )
        
        SphereXYZ_SampleStruct = z.get_bbox_moment_xyz_sample(
            Center, AverageVoxelDist2Center, Dimension_BBox_scaled
        )
        
        _, _, SphereBBoxMoment = z.calculate_bbox_moment(Voxel3D, MaxOrder, SphereXYZ_SampleStruct)
        
        [ZMoment_scaled, _] = z.calculate_bbox_moment_2_zm(
            MaxOrder,
            GCache_complex,
            GCache_pqr_linear,
            GCache_complex_index,
            CLMCache3D,
            SphereBBoxMoment
        )
        
        ZM_3DZD_invariant = z.get_3dzd_121_descriptor(ZMoment_scaled)
        
        ZM_3DZD_invariant_121 = ZM_3DZD_invariant[~np.isnan(ZM_3DZD_invariant)]
        return ZM_3DZD_invariant_121

    # Prepare random voxel - 100x100x100 to match notebook exactly
    voxel_shape = (100, 100, 100)
    Voxel3D = np.random.rand(*voxel_shape)
    logging.info(f"Initialize a {voxel_shape[0]}x{voxel_shape[1]}x{voxel_shape[2]} matrix with random values in CPU memory.")

    # Warm-up (single call)
    try:
        _ = OneTimeConversion(Voxel3D, max_order)
    except Exception as e:
        raise RuntimeError(f"OneTimeConversion failed during warm-up: {e}")
    # Timed repeated calls - matching notebook methodology exactly
    start = time.perf_counter()
    for _ in range(repeats):
        _ = OneTimeConversion(Voxel3D, max_order)
    elapsed = time.perf_counter() - start

    # Report performance (matching notebook output style)
    logging.info("\nIteratively invoke the OneTimeConversion function %d times using a for loop.", repeats)
    logging.info("Noted: Most NumPy runtimes are configured to use multithreading by default.")
    logging.info("Time elapsed is as follows:")
    logging.info("  Total time: %.2fs", elapsed)
    logging.info("  Time per iteration: %.2fms", elapsed / repeats * 1000)

    # Save detailed timing information to log file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.join(script_dir, "_simple_time_benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"benchmark_{timestamp}.log"
    log_filepath = os.path.join(benchmark_dir, log_filename)
    
    with open(log_filepath, "w") as log_file:
        log_file.write(f"ZMPY3D_JAX Simple Time Benchmark\n")
        log_file.write(f"=" * 60 + "\n")
        log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"\nConfiguration:\n")
        log_file.write(f"  Repeats: {repeats}\n")
        log_file.write(f"  Max seconds: {max_seconds}\n")
        log_file.write(f"  Max order: {max_order}\n")
        log_file.write(f"  Voxel shape: {voxel_shape}\n")
        log_file.write(f"\nResults:\n")
        log_file.write(f"  Total time: {elapsed:.4f}s\n")
        log_file.write(f"  Time per iteration: {elapsed / repeats * 1000:.4f}ms\n")
        log_file.write(f"  Iterations per second: {repeats / elapsed:.2f}\n")
        log_file.write(f"\nStatus: {'PASS' if elapsed <= max_seconds else 'FAIL'}\n")
        if elapsed > max_seconds:
            log_file.write(f"  Exceeded threshold by: {elapsed - max_seconds:.2f}s\n")
    
    logging.info(f"Benchmark results saved to: {log_filepath}")

    # Assert the elapsed time is below the threshold
    assert elapsed <= max_seconds, (
        f"Zernike pipeline exceeded allowed time: {elapsed:.2f}s > {max_seconds}s "
        f"(repeats={repeats}, voxel={voxel_shape}, max_order={max_order})"
    )