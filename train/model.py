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


class Encoder(nnx.Module):
    def __init__(self, height, width, channels, patch_size, depth,
    mlp_dim, num_heads, qkv_features, max_temporal_len,
    spatial_compression_rate, rngs: nnx.Rngs,
    dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        max_spatial_len = height // patch_size * width // patch_size


        self.last_dim = channels * patch_size * patch_size
        self.patch_embedding = PatchEmbedding(height, width, channels, patch_size, rngs,
                                              dtype=dtype, param_dtype=param_dtype)
        self.layers = []
        self.spatial_compression = nnx.Linear(self.last_dim, self.last_dim // spatial_compression_rate,
                                              dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.variance_estimator = nnx.Linear(self.last_dim, self.last_dim // spatial_compression_rate,
                                             dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.selection_layer1 = nnx.Linear(self.last_dim // spatial_compression_rate, 1,
                                           dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.selection_layer2 = nnx.Linear(max_spatial_len, 1,
                                           dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.gumbel_sigmoid = GumbelSigmoidSTE(temperature = 1.0)

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
        mean = self.spatial_compression(x)
        variance = jax.nn.softplus(self.variance_estimator(x)) # Predict softplus^-1(variance) instead of log
        log_variance = jnp.log(variance)
        selection_intermediate = self.selection_layer1(mean)
        selection_intermediate = rearrange(selection_intermediate, "b t hw 1 -> b t hw")
        selection = self.gumbel_sigmoid(self.selection_layer2(selection_intermediate) + 1, rngs, train=train)
        selection = rearrange(selection, "b t 1 -> b t 1 1")
        return mean, log_variance, selection

class Decoder(nnx.Module):
    def __init__(self, height, width, channels, patch_size, depth,
    mlp_dim, num_heads, qkv_features, max_temporal_len,
    spatial_compression_rate, unembedding_upsample_rate, rngs: nnx.Rngs,
    dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.last_dim = channels * patch_size * patch_size
        self.patch_unembedding = PatchUnEmbedding(height, width, channels, patch_size, unembedding_upsample_rate, rngs,
                                                  dtype=dtype, param_dtype=param_dtype)
        self.layers = []
        self.spatial_decompression = nnx.Linear(self.last_dim // spatial_compression_rate, self.last_dim,
                                                dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        max_spatial_len = height // patch_size * width // patch_size
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
        self.unet = UNet(channels=channels * unembedding_upsample_rate, base_features=16, num_levels=3,
                         out_features=channels, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

    def __call__(self, x: Float[Array, "b time hw ppc"], mask: Float[Array, "b 1 1 time"], rngs: nnx.Rngs, train: bool = True):
        x = self.spatial_decompression(x)
        for layer in self.layers:
            x = layer(x, mask)
        convolutional_upsampled_features, x = self.patch_unembedding(x)
        unet_output = self.unet(convolutional_upsampled_features)
        x = x + unet_output
        return x



class VideoVAE(nnx.Module):
    def __init__(self, height, width, channels, patch_size, depth,
    mlp_dim, num_heads, qkv_features, max_temporal_len,
    spatial_compression_rate, unembedding_upsample_rate, rngs: nnx.Rngs,
    dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        key = rngs.sampling()
        super().__init__()
        self.encoder = Encoder(height, width, channels, patch_size, depth,
            mlp_dim, num_heads, qkv_features, max_temporal_len,
            spatial_compression_rate, rngs, dtype=dtype, param_dtype=param_dtype)
        self.decoder = Decoder(height, width, channels, patch_size, depth,
            mlp_dim, num_heads, qkv_features, max_temporal_len,
            spatial_compression_rate, unembedding_upsample_rate, rngs,
            dtype=dtype, param_dtype=param_dtype)
        self.fill_token = nnx.Param(jax.random.normal(key, (1, 1, 1, channels * patch_size * patch_size // spatial_compression_rate)) * 0.02, trainable = True)


    def __call__(self, x: Float[Array, "b time height width channels"], mask: Float[Array, "b time 1 1"], rngs: nnx.Rngs, train: bool = True):
        #mask = rearrange(mask, "b 1 1 time -> b time 1 1")
        mean, log_variance, selection = self.encoder(x, mask, rngs, train=train)
        # Mean, logvar in shape (b, t, hw, c), selection in shape (b, t, hw, 1)

        if train:
            key = rngs.sampling()
            noise = jax.random.normal(key, log_variance.shape)
            std = jnp.exp(log_variance / 2)
            sampled_latent = mean + noise * std
        else:
            # During eval, use deterministic mean
            sampled_latent = mean

        compressed_representation = self.fill_token * (1 - selection) + sampled_latent * selection
        # selection = 1 means keep, 0 means delete
        reconstruction = self.decoder(compressed_representation, mask, rngs, train=train)
        return reconstruction, compressed_representation, selection, log_variance, mean




if __name__ == "__main__":
    # 1. Get the GPU device handle
    seed = 42
    key = jax.random.key(seed)
    try:
        gpu_device = jax.devices('gpu')[0] # 'cuda' works too, but 'gpu' is the generic backend name
    except RuntimeError:
        raise RuntimeError("No GPU found! Is JAX installed with CUDA support?")
    temporal_length = 128
    input_image = jax.random.normal(key, (10, temporal_length, 256, 256, 3)) * 0.02
    encoder = Encoder(height=256, width=256, channels=3, patch_size=16,
    depth=6, mlp_dim=512, num_heads=8, qkv_features=128,
    max_temporal_len=temporal_length, spatial_compression_rate=4, rngs = nnx.Rngs(0))



    jit_forward = nnx.jit(encoder.__call__)

    import time
    start = time.perf_counter()
    output = jit_forward(input_image)
    print(time.perf_counter() - start)
    for i in range(100):
        output = jit_forward(input_image)
    print(time.perf_counter() - start)
    params = nnx.state(encoder, nnx.Param)

    # 2. Count using standard JAX utilities
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    print(f"Trainable Parameters: {num_params / 10**6} Million")
