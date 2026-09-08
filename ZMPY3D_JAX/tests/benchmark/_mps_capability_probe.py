"""Run manually under ``JAX_PLATFORMS=mps``; emits machine-readable capabilities."""
import json
import jax
import jax.numpy as jnp


def probe(name, function):
    try:
        value = jax.block_until_ready(function())
        return {"supported": True, "shape": list(getattr(value, "shape", ())) }
    except Exception as error:  # hardware capability report
        return {"supported": False, "error": f"{type(error).__name__}: {error}"}


def main():
    results = {"backend": jax.default_backend(), "devices": [str(x) for x in jax.devices()]}
    results["complex64_conj"] = probe("complex64", lambda: jnp.conj(jnp.ones(4, dtype=jnp.complex64)))
    results["scan_2d"] = probe("scan", lambda: jax.lax.scan(lambda c, x: (c + x, c), jnp.zeros((2, 2), jnp.float32), jnp.ones((3, 2, 2), jnp.float32))[0])
    results["roll_gather_vmap"] = probe("array", lambda: jax.vmap(lambda x: jnp.roll(x, 1)[jnp.array([0, 2])])(jnp.ones((2, 4), jnp.float32)))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
