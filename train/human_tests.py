import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from einops import rearrange, repeat

#jax.config.update("jax_enable_x64", True)



from model import Encoder, VideoVAE


rng = nnx.Rngs(0)
seed = 42
key = jax.random.key(seed)
batch_size = 3
temporal_length = 11
input_image = jax.random.normal(key, (batch_size, temporal_length, 256, 256, 3)) * 0.02
input_mask = jnp.ones((batch_size, 1, 1, temporal_length), dtype=bool) 
input_mask = input_mask.at[:, :, :, 10:].set(False)
input_mask = repeat(input_mask, "b 1 1 time -> b hw 1 1 time", hw = 256 // 16 * 256 // 16)
input_mask = rearrange(input_mask, "b hw 1 1 time -> (b hw) 1 1 time")


vae = VideoVAE(height=256, width=256, channels=3, patch_size=16, 
    depth=6, mlp_dim=512, num_heads=8, qkv_features=128,
    max_temporal_len=temporal_length, spatial_compression_rate=4, unembedding_upsample_rate=1, rngs = nnx.Rngs(0))
jit_forward = nnx.jit(vae.__call__)
#jit_forward = vae.__call__
reconstruction, _, selection, _, _ = jit_forward(input_image, input_mask, rngs = rng)
print(reconstruction.shape, input_image.shape, selection.shape)

'''
reconstruction, _, _ = jit_forward(input_image, input_mask, rngs = rng, deterministic = True)
cut_input_image = input_image[:, :10, :, :, :]
cut_input_mask = jnp.ones((batch_size, 1, 1, 10), dtype=bool)
cut_input_mask = repeat(cut_input_mask, "b 1 1 time -> b hw 1 1 time", hw = 256 // 16 * 256 // 16)
cut_input_mask = rearrange(cut_input_mask, "b hw 1 1 time -> (b hw) 1 1 time")


cut_reconstruction, _, _ = jit_forward(cut_input_image, cut_input_mask, rngs = rng, deterministic = True)
print("Mask max diff: ",jnp.max(jnp.abs(reconstruction[:, :10, :, :] - cut_reconstruction)))
print("Mask mean diff: ",jnp.mean(jnp.abs(reconstruction[:, :10, :, :] - cut_reconstruction)))


input_image = jax.random.normal(key, (batch_size, temporal_length, 256, 256, 3)) * 0.02
encoder_output, _, _ = jit_forward(input_image, input_mask, rngs = rng)

unbatched_encoder_output, variance, selection = jit_forward(input_image[0:1, :, :, :, :], input_mask[0:1, :, :, :], rngs = rng)
print("Unbatched max diff: ",jnp.max(jnp.abs(unbatched_encoder_output - encoder_output[0:1, :, :, :])))
print("Unbatched mean diff: ",jnp.mean(jnp.abs(unbatched_encoder_output - encoder_output[0:1, :, :, :])))
print(selection)
assert jnp.allclose(unbatched_encoder_output, encoder_output[0:1, :, :, :], atol=1e-1)
'''
input_mask = rearrange(input_mask, "b 1 1 time -> b time 1 1")
### ENCODER TESTS ###
encoder = Encoder(height=256, width=256, channels=3, patch_size=16, 
    depth=6, mlp_dim=512, num_heads=8, qkv_features=128,
    max_temporal_len=temporal_length, spatial_compression_rate=4, rngs = nnx.Rngs(0))
#jit_forward = nnx.jit(encoder.__call__) # JIT ing encoder causes a crash, no idea why
jit_forward = encoder.__call__

### TEST MASKING ###
# batch, head, query_length, kv length
# mask over only kv length


encoder_output, variance, selection = jit_forward(input_image, input_mask, rngs = rng)


cut_input_image = input_image[:, :10, :, :, :]
cut_input_mask = jnp.ones((batch_size, 1, 1, 10), dtype=bool)
cut_input_mask = repeat(cut_input_mask, "b 1 1 time -> b hw 1 1 time", hw = 256 // 16 * 256 // 16)
cut_input_mask = rearrange(cut_input_mask, "b hw 1 1 time -> (b hw) 1 1 time")
cut_input_mask = rearrange(cut_input_mask, "b 1 1 time -> b time 1 1")
cut_encoder_output, variance, selection = jit_forward(cut_input_image, cut_input_mask, rngs = rng)
print(selection.shape)
print("Mask max diff: ",jnp.max(jnp.abs(encoder_output[:, :10, :, :] - cut_encoder_output)))
print("Mask mean diff: ",jnp.mean(jnp.abs(encoder_output[:, :10, :, :] - cut_encoder_output)))

assert jnp.allclose(encoder_output[:, :10, :, :], cut_encoder_output, atol=1e-1)


### TEST BATCH ISOLATION ###
input_image = jax.random.normal(key, (batch_size, temporal_length, 256, 256, 3)) * 0.02
encoder_output, variance, selection = jit_forward(input_image, input_mask, rngs = rng)

unbatched_encoder_output, variance, selection = encoder(input_image[0:1, :, :, :, :], input_mask[0:1, :, :, :], rngs = rng)
print("Unbatched max diff: ",jnp.max(jnp.abs(unbatched_encoder_output - encoder_output[0:1, :, :, :])))
print("Unbatched mean diff: ",jnp.mean(jnp.abs(unbatched_encoder_output - encoder_output[0:1, :, :, :])))


assert jnp.allclose(unbatched_encoder_output, encoder_output[0:1, :, :, :], atol=1e-1)


print("ALL ENCODER TESTS PASSED")



