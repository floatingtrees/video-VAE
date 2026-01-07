import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array

class Encoder(nnx.Module):
    def __init__(self, height, width, channels,rngs: nnx.Rngs):
        super().__init__()

        
    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[Array, "b time height width channels"]):
        pass







if __name__ == "__main__":
    # 1. Get the GPU device handle
    try:
        gpu_device = jax.devices('gpu')[0] # 'cuda' works too, but 'gpu' is the generic backend name
    except RuntimeError:
        print("No GPU found! Is JAX installed with CUDA support?")
        gpu_device = jax.devices('cpu')[0]
    encoder = Encoder(rngs = nnx.Rngs(0))
    nnx.display(encoder)