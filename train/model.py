import jax
import jax.numpy as jnp

# 1. Get the GPU device handle
try:
    gpu_device = jax.devices('gpu')[0] # 'cuda' works too, but 'gpu' is the generic backend name
except RuntimeError:
    print("No GPU found! Is JAX installed with CUDA support?")
    gpu_device = jax.devices('cpu')[0]