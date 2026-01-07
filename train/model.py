import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from layers import PatchEmbedding, GumbelSoftmax





class Encoder(nnx.Module):
    def __init__(self, height, width, channels, patch_size, rngs: nnx.Rngs):
        super().__init__()
        self.patch_embedding = PatchEmbedding(height, width, channels, patch_size, rngs)

    def __call__(self, x: Float[Array, "b time height width channels"]):
        return self.patch_embedding(x)







if __name__ == "__main__":
    # 1. Get the GPU device handle
    seed = 42
    key = jax.random.key(seed)
    try:
        gpu_device = jax.devices('gpu')[0] # 'cuda' works too, but 'gpu' is the generic backend name
    except RuntimeError:
        print("No GPU found! Is JAX installed with CUDA support?")
        gpu_device = jax.devices('cpu')[0]
    input_image = jax.random.normal(key, (5, 32, 256, 256, 3)) * 0.02
    encoder = Encoder(height=256, width=256, channels=128, patch_size=16, rngs = nnx.Rngs(0))
    output = encoder(input_image)
    print(output.shape)