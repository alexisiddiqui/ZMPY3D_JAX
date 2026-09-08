# Direct recurrence backend notes

## Convention derivation

The recurrence evaluates `conj(B_nlm)`, where `B` uses the orthonormal radial
factor `sqrt(2n+3)`, Jacobi `P_((n-l)/2)^(0,l+1/2)(2r²-1)`, and SciPy's
Condon–Shortley spherical harmonics. Comparing the resulting polynomial to the
committed legacy coefficient tables on unrelated point sets gives the exact
closed form

`legacy_polynomial / conj(B_nlm) = sqrt(4π/3) i^m / clm[n,l,m]`.

The factor depends only on mode indices and the analytical `clm` cache. No
fixture-derived values are stored. Cell-level float64 checks at order 6 give a
global relative L2 error below `1.1e-14`, including cells outside the unit ball.

## Quadrature

The initial experimental default is the exact rule `q = ceil((max_order+1)/2)`
(`q=11` at order 20). It is intentionally conservative pending the planned
multi-structure sweep; the 1e-5 worst-case threshold has not been relaxed.

## Sparse packing

Occupied cells are compacted on the host and padded to the maximum occupancy in
the current batch. Padding uses zero weights and finite origin coordinates.
Different occupancy maxima therefore produce different compiled shapes; fixed
occupancy buckets are the intended follow-up if recompilation is material.

## MPS

The Python 3.13 environment uses jax-mps 0.10.10 with JAX/jaxlib 0.10.2. The
capability probe succeeds for complex64 conjugation, a 2D `lax.scan` carry, and
the roll/gather/vmap combination. MPS cannot run the mixed Cartesian baseline
because that path requires float64.

Direct recurrence results on the same fixtures used for the CPU measurements:

| Order | Grid | MPS steady | CPU steady | MPS speedup | MPS compile |
|---:|---:|---:|---:|---:|---:|
| 6 | 8³ | 3.792 ms | 6.060 ms | 1.60× | 4.64 s |
| 20 | 4³ | 106.315 ms | 314.104 ms | 2.95× | 6.52 s |

Experimental `JAX_MPS_ASYNC_DISPATCH=1` did not materially change the blocked
order-20 timing (106.46 ms median); it reduced compile/startup time in that run
to 5.09 s. The synchronous setting remains the safer default.

## CPU comparison (2026-09-07)

Both approaches were JIT compiled and timed over repeated calls on the same
float32 voxel fixture at each order. Accuracy is against an untimed float64
Cartesian reference.

| Order | Grid | Mixed Cartesian steady | Direct steady | Direct / Cartesian | Mixed Cartesian rel. L2 | Direct rel. L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 8³ | 0.0575 ms | 6.060 ms | 105.5× | 1.89e-8 | 1.27e-6 |
| 20 | 4³ | 3.119 ms | 314.10 ms | 100.7× | 2.09e-8 | 3.35e-8 |

The accurate Cartesian baseline is the production `moments_x64` frontier: both
Cartesian integration and coefficient contraction use float64 before results
are cast back to complex64. At order 20 the exact direct method has comparable
accuracy but is roughly two orders of magnitude slower on CPU for this fixture.
The order-20 grid was deliberately limited to 4³ because `q=11` evaluates
1,331 quadrature points per cell. Compile times were 0.395 s Cartesian versus
15.40 s direct. These numbers characterize the current exact implementation;
the quadrature sweep and sparse-kernel specialization are the primary expected
performance levers.
