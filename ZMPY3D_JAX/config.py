import jax
import jax.numpy as jnp

global FLOAT_DTYPE
global COMPLEX_DTYPE
FLOAT_DTYPE = jnp.float32
COMPLEX_DTYPE = jnp.complex64


def configure_for_scientific_computing(
    enable_x64: bool = True,
    platform: str = "CPU",  # None = auto, 'cpu', 'gpu', 'tpu'
):
    """
    Configure JAX for scientific computing with ZMPY3D_JAX.

    Parameters
    ----------
    enable_x64 : bool, default=True
        Enable float64 precision. Critical for accurate Zernike moment calculations.
    platform : str, optional
        Force specific platform ('cpu', 'gpu', 'tpu'). None uses JAX default.

    Notes
    -----
    This function should be called ONCE at program startup, before any JAX operations.
    Float64 precision is strongly recommended for ZMPY3D_JAX to avoid numerical errors
    in iterative algorithms and accumulations.

    Examples
    --------
    >>> import ZMPY3D_JAX as z
    >>> z.configure_for_scientific_computing()  # Recommended
    >>> # Now use the library...
    """

    global FLOAT_DTYPE
    global COMPLEX_DTYPE
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
        print("JAX configured for float64 precision")
        # set the global dtype for arrays to float64
        FLOAT_DTYPE = jnp.float64
        COMPLEX_DTYPE = jnp.complex128
    else:
        jax.config.update("jax_enable_x64", False)
        print(
            "Warning: JAX float64 precision is disabled. This may lead to numerical inaccuracies."
        )
        FLOAT_DTYPE = jnp.float32
        COMPLEX_DTYPE = jnp.complex64

    if platform is not None:
        jax.config.update("jax_platform_name", platform.lower())
        print(f"JAX configured for platform: {platform}")
