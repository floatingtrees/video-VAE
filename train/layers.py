import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from einops import rearrange

class PatchEmbedding(nnx.Module):
    def __init__(self, height, width, channels, patch_size, rngs: nnx.Rngs):
        super().__init__()
        self.patch_size = patch_size
        
        self.linear = nnx.Linear(patch_size * patch_size, patch_size * patch_size, rngs = rngs)
        self.norm = nnx.LayerNorm(patch_size * patch_size, rngs = rngs)
        
    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[Array, "b time height width channels"]):
        x = rearrange(x, "b t (h p1) (w p2) c -> b t c (h w) (p1 p2)",
            p1 = self.patch_size, p2 = self.patch_size)
        x = self.linear(x)
        x = self.norm(x)
        return x

class GumbelSoftmax(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        super().__init__()
        
    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[Array, "b time height width channels"]):
        return x