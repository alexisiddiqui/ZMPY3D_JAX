# TensorFlow-to-JAX companion-solver optimization

## Cross-framework benchmark

The pinned TensorFlow implementation is benchmarked by
`ZMPY3D_JAX/tests/benchmark/test_benchmark_tensorflow.py` in a separate worker process and a
dedicated environment at `ZMPY3D_JAX/tests/benchmark/tensorflow_env`. The environment pins
TensorFlow `2.20.0`, its CUDA extra, TensorFlow Probability `0.25.0` with the `tf` extra, and
Python 3.13. The submodule itself remains unmodified.

The benchmark compares production JAX mixed precision and x64-matched JAX against TensorFlow’s
float64 implementation. It records warmed core and public end-to-end timings for CPU/GPU, orders
6/20, modes, and workload sizes. TensorFlow’s generator is per-structure while JAX uses padded
true batches, so core and end-to-end measurements are labeled separately. The order-6 GPU smoke
path passed both views; x64 descriptor parity passed at the existing `5e-5/5e-6` tolerance.

## Reference and exactness decision

The design reference is `tawssie/ZMPY3D_TF` at revision
[`bb57903716c9eb7666ca0bfcfb3af6135a2d63cd`](https://github.com/tawssie/ZMPY3D_TF/tree/bb57903716c9eb7666ca0bfcfb3af6135a2d63cd).
The relevant sources are its
[single companion solver](https://github.com/tawssie/ZMPY3D_TF/blob/bb57903716c9eb7666ca0bfcfb3af6135a2d63cd/ZMPY3D_TF/lib/eigen_root_tf.py),
[batched companion solver](https://github.com/tawssie/ZMPY3D_TF/blob/bb57903716c9eb7666ca0bfcfb3af6135a2d63cd/ZMPY3D_TF/lib/eigen_root_tf2.py),
[rotation kernel](https://github.com/tawssie/ZMPY3D_TF/blob/bb57903716c9eb7666ca0bfcfb3af6135a2d63cd/ZMPY3D_TF/lib/calculate_zm_by_ab_rotation01_tf.py), and
[prefetched voxel dataset](https://github.com/tawssie/ZMPY3D_TF/blob/bb57903716c9eb7666ca0bfcfb3af6135a2d63cd/ZMPY3D_TF/lib/pdb_2_voxel_dataset02.py).

“Exact” here means retaining the original companion-matrix polynomial formulation; eigenvalue
ordering is not part of the contract. All production primary roots use companion eigensolves.
The already accepted exact factorization of the secondary quartic is retained, including its two
compact useful-real-root slots. No analytic quartic is introduced. The stable analytic odd-order
quadratic remains available only as an explicitly named benchmark experiment.

## TensorFlow-to-JAX mapping

| TensorFlow idea | JAX implementation |
| --- | --- |
| Construct companions for a fixed degree and solve them together | `batched_eigen_root` accepts arbitrary leading axes and calls one batched `jnp.linalg.eigvals` |
| Group work having the same matrix shape | Odd orders 3/5 form a degree-two group; even orders 2/4 form a degree-four group |
| Evaluate rotations across candidates | The existing nested kernel remains production; `_calculate_rotation_flat_batch` is the candidate-parallel prototype |
| Preserve formula term order | The flattened prototype reuses the same log-domain kernel and segmented reduction without reordering terms |
| Overlap dataset production with device consumption | `_run_prepared_batch(..., prefetch=True)` uses two preparation workers and at most two pending padded chunks |

Leading-zero polynomials still return fixed-size all-NaN root arrays. Candidate masks, identity
handling, descriptor masks, and public APIs are unchanged.

## Baseline

The last clean schema-v6 RTX 3090 batch-16 profile recorded 1.086 ms/protein for the separately
launched device core, 1.010 ms/protein for whole-pipeline JIT, 0.489 ms/protein for compact
candidate generation, 0.468 ms/protein for rotation, and 6.997 ms/protein prepared end to end.
CPU batch-16 whole-JIT was 64.892 ms/protein. The accepted order-20 6NT5/6NT6 values were score
error `0.0029475`, difference cosine `0.9997608`, and separation ratio `0.9994425`.

## Phase 1: device layout

Implemented:

- arbitrary-leading-dimension, fixed-degree companion batches;
- companion defaults for single, batched, fused, and staged/parity normalization paths;
- benchmark-visible per-order and degree-grouped candidate layouts;
- nested and flattened rotation variants; and
- whole-JIT composition of grouped roots with nested rotation and with flattened rotation.

Two clean RTX 3090 schema-v7 runs (9 samples, 3 repeats) measured per-order companion normalization
at `1.3777` and `1.3747` ms/protein and degree-grouped companion normalization at `1.2873` and
`1.2817` ms/protein. The corresponding throughput gains were `7.02%` and `7.26%`, passing the 5%
gate twice. Batch-2 grouping also improved throughput by `8.80%` and `6.64%`. The CPU batch-16
comparison was neutral (`74.639` versus `74.734` ms/protein, a `0.13%` slowdown), safely inside the
10% regression guard; CPU batch 2 improved by `12.3%`. Degree grouping is therefore approved for
production promotion with nested rotation.

Flattened rotation alone was `2.15%` and `2.60%` slower on GPU batch 16 and approximately `98%`
slower on CPU. Although the grouped-plus-flattened whole-JIT variant cleared 5% relative to the
per-order baseline, it was slower than grouped-plus-nested in both GPU runs. Flattening remains a
benchmark-only prototype.

## Phase 2: host prefetch

The internal benchmark now compares sequential preparation with a bounded prefetch producer.
It preserves path order, raises worker exceptions on the consumer thread, skips threading for a
one-chunk workload, and holds no more than two pending padded chunks. The CLI signature and
production sequential default are unchanged.

GPU batch-16 prefetch gains were only `0.37%` and `1.38%`; CPU batch-16 gained `0.71%`. These are
well below the 10% gate, so prefetch remains benchmark-only and sequential preparation remains the
production default.

## Promotion accuracy results

The focused CUDA suite passed all six order-6/order-20 accuracy and five-run determinism tests.
For order-20 6NT5/6NT6, score error was `0.00294755`, difference cosine was `0.99976075`, and the
separation ratio was `0.99944251`. Candidate counts remained `8/4/8/4`. These satisfy every
numerical promotion gate.

## Acceptance gates

Root comparisons use unordered-set distance at most `1e-5` in float32 and `1e-10` in x64, with
normalized residuals at most `1e-4` and `1e-10`, respectively. Masks, counts, degenerate behavior,
and outputs must match. Float32 CUDA descriptors must be bitwise repeatable across five direct and
whole-JIT executions. Order-20 6NT5/6NT6 must retain score error at most `0.01`, difference cosine
at least `0.9995`, and separation ratio in `[0.995, 1.005]`. Batch-2 GPU, CPU batch-16, and peak
device memory regressions are capped at 5%, 10%, and 25%.

Isolated batch-16 CUDA allocator measurements recorded `303,990,784` bytes peak for per-order
companion normalization and `378,566,400` bytes for degree grouping, a `24.53%` increase that
passes the 25% gate. All promotion gates are satisfied, and degree-grouped companion normalization
with nested rotation is the production whole-pipeline default.

## Cross-framework benchmark results

The full matrix used the pinned TensorFlow revision, JAX 0.11.0, TensorFlow 2.20.0, an Intel
i7-12700KF CPU, and the RTX 3090 CUDA device used for the preceding promotion measurements. Every
cell contains two fresh-process trials with three synchronized post-warmup samples per trial. Values
below are medians of the two trial medians. A speedup above 1 means JAX is faster.

### Reference-core throughput

TensorFlow invokes its pinned core once per structure; JAX invokes one padded batch. These values
therefore measure each reference implementation's core throughput, including its batching strategy.

| Device | Order | Batch | JAX production ms/protein | JAX x64 ms/protein | TensorFlow x64 ms/protein | Production speedup | x64 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 6 | 1 | 1.175 | 1.018 | 3.690 | 3.14x | 3.62x |
| CPU | 6 | 2 | 1.007 | 1.158 | 3.519 | 3.50x | 3.04x |
| CPU | 6 | 16 | 1.304 | 1.842 | 3.173 | 2.43x | 1.72x |
| CPU | 20 | 1 | 73.982 | 27.695 | 41.697 | 0.56x | 1.51x |
| CPU | 20 | 2 | 65.963 | 26.785 | 38.868 | 0.59x | 1.45x |
| CPU | 20 | 16 | 68.047 | 27.041 | 39.400 | 0.58x | 1.46x |
| GPU | 6 | 1 | 1.377 | 11.796 | 5.037 | 3.66x | 0.43x |
| GPU | 6 | 2 | 1.027 | 11.432 | 3.883 | 3.78x | 0.34x |
| GPU | 6 | 16 | 0.786 | 10.030 | 4.235 | 5.39x | 0.42x |
| GPU | 20 | 1 | 2.097 | 167.923 | 8.497 | 4.05x | 0.05x |
| GPU | 20 | 2 | 1.753 | 167.689 | 7.772 | 4.43x | 0.05x |
| GPU | 20 | 16 | 1.299 | 169.396 | 8.968 | 6.90x | 0.05x |

Production JAX is the relevant deployment comparison: it leads TensorFlow by 2.43-3.50x on CPU at
order 6 and 3.66-6.90x on GPU. CPU order-20 production mixed precision was slower than JAX x64 in
this matrix. The diagnostic JAX x64 CUDA view is 2.3x slower at order 6
and about 19-20x slower at order 20; it is a parity configuration, not the production strategy.

### Public workflow throughput

These timings include PDB parsing, voxelization, preparation, public API dispatch, transfers, and
output synchronization. TensorFlow's per-structure workflow and JAX's padded-batch CLI are materially
different, so this table describes current application workflows rather than isolated kernel speed.

| Device | Order | Batch | JAX production ms/protein | JAX x64 ms/protein | TensorFlow ms/protein |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 6 | 1 | 1206.79 | 678.00 | 566.61 |
| CPU | 6 | 2 | 622.55 | 359.32 | 283.69 |
| CPU | 6 | 16 | 88.87 | 52.28 | 43.46 |
| CPU | 20 | 1 | 1959.65 | 854.26 | 615.14 |
| CPU | 20 | 2 | 1052.73 | 453.19 | 320.41 |
| CPU | 20 | 16 | 206.04 | 83.80 | 69.19 |
| GPU | 6 | 1 | 1385.64 | 1131.66 | 583.46 |
| GPU | 6 | 2 | 720.61 | 563.62 | 295.35 |
| GPU | 6 | 16 | 95.82 | 90.50 | 47.27 |
| GPU | 20 | 1 | 2172.43 | 1598.97 | 635.57 |
| GPU | 20 | 2 | 1092.48 | 942.18 | 332.82 |
| GPU | 20 | 16 | 151.23 | 332.82 | 70.05 |

### Cross-framework accuracy

The table reports the worst metric over both trials and all three batch sizes. All x64 cells passed
`rtol=5e-5`, `atol=5e-6` for every descriptor row. Large production maximum absolute errors occur in
high-magnitude descriptor elements; cosine similarity remains above `0.99999`, and the established
order-20 structure-level promotion gates remain the authoritative production accuracy check.

| Device | Order | JAX view | Max absolute error | Max RMSE | Minimum cosine |
| --- | ---: | --- | ---: | ---: | ---: |
| CPU | 6 | production | 0.109 | 0.0201 | 0.999999481 |
| CPU | 6 | x64 | 1.08e-11 | 1.38e-12 | 1.000000000 |
| CPU | 20 | production | 0.138 | 0.0217 | 0.999991751 |
| CPU | 20 | x64 | 8.36e-9 | 2.46e-10 | 1.000000000 |
| GPU | 6 | production | 0.156 | 0.0300 | 0.999998868 |
| GPU | 6 | x64 | 1.08e-11 | 1.37e-12 | 1.000000000 |
| GPU | 20 | production | 0.163 | 0.0230 | 0.999990762 |
| GPU | 20 | x64 | 2.85e-8 | 5.29e-10 | 1.000000000 |

Raw schema-v2 reports were written to `/tmp/zmpy3d_tf_full_cpu` and
`/tmp/zmpy3d_tf_full_gpu` for this run. The benchmark defaults to the ignored repository report
directory for subsequent reproducible runs.

The follow-up order-20 stage profile and isolated CUDA worker trace are documented in
`plans/PERFORMANCE_HANDOFF.md`. They identify order-2/order-4 candidate generation and ZM rotation
as the next optimization target while preserving companion eigensolves as the production root
strategy. An attempted mask-aware rotation prototype with CPU staged dispatch and a fixed-shape
whole-JIT variant was closed: the former is outside the promoted executable and the latter cannot
skip `vmap` lanes. The dense grouped companion path remains the sole production and benchmark
baseline.
