from quopri import encodestring
from timeit import default_timer
import jax
#jax.config.update("jax_numpy_rank_promotion", "warn")
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from layers import PatchEmbedding, FactoredAttention, GumbelSigmoidSTE, PatchUnEmbedding
from einops import rearrange
from unet import UNet


class Classifier(nnx.Module):
    def __init__(self, height, width, channels, patch_size, depth,
    mlp_dim, num_heads, qkv_features, max_temporal_len, rngs: nnx.Rngs,
    dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        max_spatial_len = height // patch_size * width // patch_size


        self.last_dim = channels * patch_size * patch_size
        self.patch_embedding = PatchEmbedding(height, width, channels, patch_size, rngs,
                                              dtype=dtype, param_dtype=param_dtype)
        self.layers = []
        self.classifier = nnx.Linear(self.last_dim, 1, rngs=rngs)
        for _ in range(depth):
            self.layers.append(FactoredAttention(mlp_dim = mlp_dim,
                in_features = self.last_dim,
                num_heads = num_heads,
                qkv_features = qkv_features,
                max_temporal_len = max_temporal_len,
                max_spatial_len = max_spatial_len,
                rngs = rngs,
                dtype=dtype,
                param_dtype=param_dtype
            ))

    def __call__(self, x: Float[Array, "b time height width channels"], mask: Float[Array, "b 1 1 time"], train: bool = True):
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.classifier(x)[:, 0, 0]

if __name__ == "__main__":
    model = Encoder(height=256, width=256, channels=3, patch_size=16, depth=12, mlp_dim=1024, num_heads=16, qkv_features=128, max_temporal_len=16, rngs=nnx.Rngs(1))
    x = jnp.ones((3, 16, 256, 256, 3))
    mask = jnp.zeros((3 * 256, 16, 1, 1), dtype=bool)
    print(model(x, mask).shape)