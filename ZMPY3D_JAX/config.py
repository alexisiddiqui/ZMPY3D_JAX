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
        Select float64 library defaults. When false, explicit x64 operations remain enabled
        for the automatic high-order mixed-precision moment frontier.
    platform : str, optional
        Force specific platform ('cpu', 'gpu', 'tpu'). None uses JAX default.

    Notes
    -----
    This function should be called ONCE at program startup, before any JAX operations.
    Float64 remains the recommended general-purpose scientific configuration. Float32
    order-20 batched descriptors automatically retain x64 only for sensitive moment stages.

    Examples
    --------
    >>> import ZMPY3D_JAX as z
    >>> z.configure_for_scientific_computing()  # Recommended
    >>> # Now use the library...
    """

    global FLOAT_DTYPE
    global COMPLEX_DTYPE
    platform_name = platform.lower() if platform is not None else None
    if platform_name == "mps":
        if enable_x64:
            raise ValueError("the MPS backend does not support enable_x64=True")
        jax.config.update("jax_enable_x64", False)
        FLOAT_DTYPE = jnp.float32
        COMPLEX_DTYPE = jnp.complex64
        jax.config.update("jax_platform_name", "mps")
        print("JAX configured for float32 precision on platform: mps")
        return
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
        print("JAX configured for float64 precision")
        # set the global dtype for arrays to float64
        FLOAT_DTYPE = jnp.float64
        COMPLEX_DTYPE = jnp.complex128
    else:
        # Keep float32 as the library default while permitting the order-20 mixed
        # moment frontier to issue explicit float64/complex128 operations.
        jax.config.update("jax_enable_x64", True)
        print(
            "JAX configured for float32 defaults with mixed-precision moments enabled"
        )
        FLOAT_DTYPE = jnp.float32
        COMPLEX_DTYPE = jnp.complex64

    if platform is not None:
        jax.config.update("jax_platform_name", platform_name)
        print(f"JAX configured for platform: {platform}")
