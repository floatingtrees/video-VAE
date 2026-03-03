from quopri import encodestring
from timeit import default_timer
import jax
#jax.config.update("jax_numpy_rank_promotion", "warn")
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array, Int, Bool
from layers import PatchEmbedding, FactoredAttention, GumbelSigmoidSTE, PatchUnEmbedding
from einops import rearrange
from unet import UNet
from einops import repeat
from shift_indices import shift_indices_to_left, convert_to_indices

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
        layers = []
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
            layers.append(FactoredAttention(mlp_dim = mlp_dim,
                in_features = self.last_dim,
                num_heads = num_heads,
                qkv_features = qkv_features,
                max_temporal_len = max_temporal_len,
                max_spatial_len = max_spatial_len,
                rngs = rngs,
                dtype=dtype,
                param_dtype=param_dtype
            ))
        self.layers = layers

    def __call__(self, x: Float[Array, "b time height width channels"], mask: Float[Array, "b 1 1 time"], rngs: nnx.Rngs, train: bool = True):
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        mean = self.spatial_compression(x)
        variance = jax.nn.softplus(self.variance_estimator(x).astype(jnp.float32))
        variance = (variance + 1e-6).astype(mean.dtype)
        selection_intermediate = self.selection_layer1(mean)
        selection_intermediate = rearrange(selection_intermediate, "b t hw 1 -> b t hw")
        selection = jax.nn.sigmoid(self.selection_layer2(selection_intermediate) + 1)
        return mean, variance, selection

class Decoder(nnx.Module):
    def __init__(self, height, width, channels, patch_size, depth,
    mlp_dim, num_heads, qkv_features, max_temporal_len,
    spatial_compression_rate, unembedding_upsample_rate, rngs: nnx.Rngs,
    dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.last_dim = channels * patch_size * patch_size
        self.patch_unembedding = PatchUnEmbedding(height, width, channels, patch_size, unembedding_upsample_rate, rngs,
                                                  dtype=dtype, param_dtype=param_dtype)
        layers = []
        self.spatial_decompression = nnx.Linear(self.last_dim // spatial_compression_rate, self.last_dim,
                                                dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        max_spatial_len = height // patch_size * width // patch_size
        for _ in range(depth):
            layers.append(FactoredAttention(mlp_dim = mlp_dim,
                in_features = self.last_dim,
                num_heads = num_heads,
                qkv_features = qkv_features,
                max_temporal_len = max_temporal_len,
                max_spatial_len = max_spatial_len,
                rngs = rngs,
                dtype=dtype,
                param_dtype=param_dtype
            ))
        self.layers = layers
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
    def __init__(self, height, width, channels, patch_size, encoder_depth, decoder_depth,
    mlp_dim, num_heads, qkv_features, max_temporal_len,
    spatial_compression_rate, unembedding_upsample_rate, rngs: nnx.Rngs,
    dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        key = rngs.sampling()
        super().__init__()
        self.encoder = Encoder(height, width, channels, patch_size, encoder_depth,
            mlp_dim, num_heads, qkv_features, max_temporal_len,
            spatial_compression_rate, rngs, dtype=dtype, param_dtype=param_dtype)
        self.decoder = Decoder(height, width, channels, patch_size, decoder_depth,
            mlp_dim, num_heads, qkv_features, max_temporal_len,
            spatial_compression_rate, unembedding_upsample_rate, rngs,
            dtype=dtype, param_dtype=param_dtype)
        self.fill_token = nnx.Param(jax.random.normal(key, (1, 1, 1, channels * patch_size * patch_size // spatial_compression_rate)) * 0.02, trainable = True)
        


    def __call__(self, x: Float[Array, "b time height width channels"], mask: Float[Array, "b 1 1 time"], rngs: nnx.Rngs, train: bool = True, p: int = 2):
        #mask = rearrange(mask, "b 1 1 time -> b time 1 1")
        mean, variance, selection = self.encoder(x, mask, rngs, train=train)
        # Mean, variance in shape (b, t, hw, c), selection in shape (b, t, hw, 1)

        if train:
            key = rngs.sampling()
            noise = jax.random.normal(key, variance.shape)
            std = jnp.sqrt(variance)
            sampled_latent = mean + noise * std
        else:
            # During eval, use deterministic mean
            sampled_latent = mean

        


        selection = repeat(selection, "b t 1 -> (b p) t 1 1", p = p)
        sampled_latent = repeat(sampled_latent, "b ... -> (b p) ...", p = p)
        mean = repeat(mean, "b ... -> (b p) ...", p = p)
        variance = repeat(variance, "b ... -> (b p) ...", p = p)
        mask = repeat(mask, "b ... -> (b p) ...", p = p)
        key = rngs.sampling()
        selection_mask = jax.random.bernoulli(key, p=selection).astype(sampled_latent.dtype)

        compressed_representation = self.fill_token * (1 - selection_mask) + sampled_latent * selection_mask
        # selection = 1 means keep, 0 means delete
        reconstruction = self.decoder(compressed_representation, mask, rngs, train=train)
        return reconstruction, compressed_representation, selection, selection_mask, variance, mean


    def compress(self, x: Float[Array, "b time height width channels"], mask: Float[Array, "b 1 1 time"], rngs: nnx.Rngs, train: bool = True):
        mean, variance, selection_probs = self.encoder(x, mask, rngs, train=train)
        key = rngs.sampling()
        noise = jax.random.normal(key, variance.shape)
        std = jnp.sqrt(variance)
        sampled_latent = mean + noise * std
        

        key = rngs.sampling()
        selection_mask = jax.random.bernoulli(key, p=selection_probs)
        selection_mask = rearrange(selection_mask, "b t 1 -> b t")

        batched_convert_to_indices = jax.vmap(convert_to_indices)
        selection_indices, dynamic_len = batched_convert_to_indices(selection_mask)
        batched_shift = jax.vmap(shift_indices_to_left)                                                                                                              
        compressed, compression_mask = batched_shift(sampled_latent, selection_indices, dynamic_len)

        '''
        print(selection_indices.shape, dynamic_len, compressed.shape)
        print(selection_mask)
        print("?????")
        print(selection_indices)
        print(compression_mask)
        print(jnp.allclose(compressed[0, 0], sampled_latent[0, 1]))
        '''
        return compressed, selection_indices, compression_mask

    def decompress(self, compressed: Float[Array, "b t hw d"], attention_mask: Float[Array, "b 1 1 time"],
    selection_indices: Int[Array, "b t"], compression_mask: Bool[Array, "b t"], rngs: nnx.Rngs, train: bool = True):
        b, t, hw, d = compressed.shape
        fill = rearrange(self.fill_token.value, "1 1 1 d -> 1 1 d")

        def unpack_single(compressed_single, indices, mask):
            safe_indices = jnp.where(mask, indices, 0)
            valid_data = jnp.where(mask[:, None, None], compressed_single, 0.0)
            result = jnp.zeros_like(compressed_single).at[safe_indices].add(valid_data)

            full_mask = jnp.zeros(t, dtype=bool).at[safe_indices].max(mask)
            result = jnp.where(full_mask[:, None, None], result, fill)
            return result

        full_representation = jax.vmap(unpack_single)(compressed, selection_indices, compression_mask)
        reconstruction = self.decoder(full_representation, attention_mask, rngs, train=train)
        return reconstruction


if __name__ == "__main__":
    # 1. Get the GPU device handle
    seed = 42
    key = jax.random.key(seed)
    try:
        gpu_device = jax.devices('gpu')[0] # 'cuda' works too, but 'gpu' is the generic backend name
    except RuntimeError:
        raise RuntimeError("No GPU found! Is JAX installed with CUDA support?")
    temporal_length = 5
    input_image = jax.random.normal(key, (2, temporal_length, 256, 256, 3)) * 0.02
    VAE = VideoVAE(height=256, width=256, channels=3, patch_size=16,
    encoder_depth=9, decoder_depth=12, mlp_dim=1536, num_heads=8, qkv_features=512,
    max_temporal_len=temporal_length, spatial_compression_rate=8, unembedding_upsample_rate=4, rngs = nnx.Rngs(0))
    attn_mask = jnp.ones((2, 1, 1, temporal_length), dtype=bool)

    compressed_representation, selection_indices, compression_mask = VAE.compress(input_image, attn_mask, rngs = nnx.Rngs(0))
    print(compressed_representation.shape)
    reconstruction = VAE.decompress(compressed_representation, attn_mask, selection_indices, compression_mask, rngs = nnx.Rngs(0))
    T_reconstruction, T_compressed_representation, selection, selection_mask, variance, mean = VAE(input_image, attn_mask, nnx.Rngs(0), p=1)

    
    print(jnp.max(jnp.abs(T_reconstruction - reconstruction)))

    # Test with jit
    jit_compress = nnx.jit(VAE.compress)
    jit_decompress = nnx.jit(VAE.decompress)
    jit_forward = nnx.jit(VAE.__call__, static_argnames=("p",))

    compressed_jit, indices_jit, mask_jit = jit_compress(input_image, attn_mask, rngs=nnx.Rngs(0))
    reconstruction_jit = jit_decompress(compressed_jit, attn_mask, indices_jit, mask_jit, rngs=nnx.Rngs(0))
    #T_reconstruction_jit, T_compressed_jit, selection_jit, selection_mask_jit, variance_jit, mean_jit = jit_forward(input_image, attn_mask, nnx.Rngs(0), p=1)

    print("jit compress vs eager compress:", jnp.max(jnp.abs(compressed_jit - compressed_representation)))
    print("jit decompress vs eager decompress:", jnp.max(jnp.abs(reconstruction_jit - reconstruction)))
    #print("jit forward vs eager forward:", jnp.max(jnp.abs(T_reconstruction_jit - T_reconstruction)))
    #print("jit compress->decompress vs jit forward:", jnp.max(jnp.abs(T_reconstruction_jit - reconstruction_jit)))
