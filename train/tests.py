import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from einops import rearrange

from model import Encoder

seed = 42
key = jax.random.key(seed)
batch_size = 10
### TEST MASKING ###
temporal_length = 128
input_image = jax.random.normal(key, (batch_size, temporal_length, 256, 256, 3)) * 0.02
encoder = Encoder(height=256, width=256, channels=3, patch_size=16, 
    depth=6, mlp_dim=512, num_heads=8, qkv_features=128,
    max_temporal_len=temporal_length, spatial_compression_rate=4, rngs = nnx.Rngs(0))

# batch, head, query_length, kv length
# mask over only kv length
jit_forward = nnx.jit(encoder.__call__)

input_mask = jnp.zeros((batch_size, 1, 1, temporal_length)) 
input_mask = input_mask.at[:, :, :, :10].set(float('-inf'))

encoder_output = jit_forward(input_image, input_mask)
print(encoder_output.shape)