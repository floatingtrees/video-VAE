from quopri import encodestring
import jax
#jax.config.update("jax_numpy_rank_promotion", "warn")
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from layers import PatchEmbedding, FactoredAttention, GumbelSigmoidSTE




class Encoder(nnx.Module):
    def __init__(self, height, width, channels, patch_size, depth, 
    mlp_dim, num_heads, qkv_features, max_temporal_len, 
    spatial_compression_rate, rngs: nnx.Rngs):
        super().__init__()
        self.last_dim = channels * patch_size * patch_size
        self.patch_embedding = PatchEmbedding(height, width, channels, patch_size, rngs)
        self.layers = []
        self.spatial_compression = nnx.Linear(self.last_dim, self.last_dim // spatial_compression_rate, rngs = rngs)
        self.variance_estimator = nnx.Linear(self.last_dim, self.last_dim // spatial_compression_rate, rngs = rngs)
        self.selection_layer = nnx.Linear(self.last_dim // spatial_compression_rate, 1, rngs = rngs)
        self.gumbel_sigmoid = GumbelSigmoidSTE(temperature = 1.0)
        
        max_spatial_len = height // patch_size * width // patch_size
        for _ in range(depth):
            self.layers.append(FactoredAttention(mlp_dim = mlp_dim, 
                in_features = self.last_dim,
                num_heads = num_heads,
                qkv_features = qkv_features,
                max_temporal_len = max_temporal_len,
                max_spatial_len = max_spatial_len,
                rngs = rngs
            ))

    def __call__(self, x: Float[Array, "b time height width channels"], mask: Float[Array, "b 1 1 time"], rngs: nnx.Rngs):
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        mean = self.spatial_compression(x)
        log_variance = self.variance_estimator(x)
        selection = self.gumbel_sigmoid(self.selection_layer(mean), rngs)
        return mean, log_variance, selection



def VideoVAE(nnx.Module):
    def __init__(self, height, width, channels, patch_size, depth, 
    mlp_dim, num_heads, qkv_features, max_temporal_len, 
    spatial_compression_rate, rngs: nnx.Rngs):
        key = rngs.sampling()
        super().__init__()
        self.encoder = Encoder(height, width, channels, patch_size, depth, 
            mlp_dim, num_heads, qkv_features, max_temporal_len, 
            spatial_compression_rate, rngs)
        self.decoder = None
        self.fill_token = nnx.Param(jax.random.normal(key, (1, 1, 1, channels * patch_size * patch_size)), trainable = False)
        
    def __call__(self, x: Float[Array, "b time height width channels"], mask: Float[Array, "b 1 1 time"], rngs: nnx.Rngs):
        mean, log_variance, selection = self.encoder(x, mask, rngs)
        # Mean, logvar in shape (b, t, hw, c), selection in shape (b, t, hw, 1)
        key = rngs.sampling()
        eps = 1e-20
        noise = jax.random.normal(key, log_variance.shape)
        variance = jnp.exp(log_variance)
        sampled_latents = mean + noise * jnp.sqrt(variance)






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