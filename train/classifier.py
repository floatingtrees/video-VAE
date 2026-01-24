from layers import PatchEmbedding, FactoredAttention
import jax.numpy as jnp
from flax import nnx
import jax
from jaxtyping import jaxtyped, Float, Array


class Classifier(nnx.Module):
    def __init__(self, height, width, channels, patch_size, depth,
    mlp_dim, num_heads, qkv_features, max_temporal_len,
    rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        max_spatial_len = height // patch_size * width // patch_size


        self.last_dim = channels * patch_size * patch_size
        self.patch_embedding = PatchEmbedding(height, width, channels, patch_size, rngs,
                                              dtype=dtype, param_dtype=param_dtype)
        self.layers = []
        self.binary_classifier = nnx.Linear(self.last_dim, 1,
                                              dtype=dtype, param_dtype=param_dtype, rngs=rngs)
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

    def __call__(self, x: Float[Array, "b time height width channels"], mask: Float[Array, "b 1 1 time"], rngs: nnx.Rngs, train: bool = True):
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        binary_classifier_output = self.binary_classifier(x)
        output_value = binary_classifier_output[:, 0, 0, 0] # we want 1 per sample
        # Also, we have dense attention, so we get away with checking the first element (to avoid masking issues)
        return output_value

if __name__ == "__main__":
    height, width = (256, 256)
    patch_size = 16
    channels = 3
    depth = 6
    mlp_dim = 512
    num_heads = 8
    qkv_features = 128
    max_temporal_len = 128
    spatial_compression_rate = 4
    rngs = nnx.Rngs(0)
    classifier = Classifier(height, width, channels, patch_size, depth, mlp_dim, num_heads, qkv_features, max_temporal_len, rngs)
    temporal_length = 128
    key = jax.random.key(0)
    input_image = jax.random.normal(key, (10, temporal_length, 256, 256, 3)) * 0.02
    input_mask = jnp.ones((10 * 256, 1, 1, temporal_length), dtype=bool)
    print(classifier(input_image, input_mask, rngs, train=True).shape)