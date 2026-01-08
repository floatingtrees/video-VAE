from quopri import encodestring
import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from layers import PatchEmbedding, GumbelSoftmaxSTE, FactoredAttention





class Encoder(nnx.Module):
    def __init__(self, height, width, channels, patch_size, depth, 
    mlp_dim, num_heads, qkv_features, max_temporal_len, 
    spatial_compression_rate, rngs: nnx.Rngs):
        super().__init__()
        self.last_dim = channels * patch_size * patch_size
        self.patch_embedding = PatchEmbedding(height, width, channels, patch_size, rngs)
        self.layers = []
        self.spatial_compression = nnx.Linear(self.last_dim, self.last_dim // spatial_compression_rate, rngs = rngs)
        
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


    def __call__(self, x: Float[Array, "b time height width channels"]):
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.spatial_compression(x)
        return x







if __name__ == "__main__":
    # 1. Get the GPU device handle
    seed = 42
    key = jax.random.key(seed)
    try:
        gpu_device = jax.devices('gpu')[0] # 'cuda' works too, but 'gpu' is the generic backend name
    except RuntimeError:
        print("No GPU found! Is JAX installed with CUDA support?")
        gpu_device = jax.devices('cpu')[0]
    temporal_length = 1024
    input_image = jax.random.normal(key, (5, temporal_length, 256, 256, 3)) * 0.02
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