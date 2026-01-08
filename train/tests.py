import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from einops import rearrange, repeat

from model import Encoder

seed = 42
key = jax.random.key(seed)
batch_size = 3
### TEST MASKING ###
temporal_length = 128
input_image = jax.random.normal(key, (batch_size, temporal_length, 256, 256, 3)) * 0.02
encoder = Encoder(height=256, width=256, channels=3, patch_size=16, 
    depth=6, mlp_dim=512, num_heads=8, qkv_features=128,
    max_temporal_len=temporal_length, spatial_compression_rate=4, rngs = nnx.Rngs(0))
jit_forward = nnx.jit(encoder.__call__)

#jit_forward = encoder.__call__


# batch, head, query_length, kv length
# mask over only kv length


input_mask = jnp.ones((batch_size, 1, 1, temporal_length), dtype=bool) 
input_mask = input_mask.at[:, :, :, 10:].set(False)
input_mask = repeat(input_mask, "b 1 1 time -> b hw 1 1 time", hw = 256 // 16 * 256 // 16)
input_mask = rearrange(input_mask, "b hw 1 1 time -> (b hw) 1 1 time")

encoder_output = jit_forward(input_image, input_mask)
print(encoder_output.shape)


cut_input_image = input_image[:, :10, :, :, :]
cut_input_mask = jnp.ones((batch_size, 1, 1, 10), dtype=bool)
cut_input_mask = repeat(cut_input_mask, "b 1 1 time -> b hw 1 1 time", hw = 256 // 16 * 256 // 16)
cut_input_mask = rearrange(cut_input_mask, "b hw 1 1 time -> (b hw) 1 1 time")

cut_encoder_output = jit_forward(cut_input_image, cut_input_mask)

print(jnp.max(jnp.abs(encoder_output[:, :10, :, :] - cut_encoder_output)))
assert jnp.allclose(encoder_output[:, :10, :, :], cut_encoder_output)