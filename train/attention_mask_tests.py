import jax
from jax import numpy as jnp


b = 17
h = 19
s = 15
d = 13
q = jax.random.normal(shape=(b, s, h, d), key = jax.random.key(1))
k = jax.random.normal(shape=(b, s, h, d), key=jax.random.key(2))
v = jax.random.normal(shape=(b, s, h, d), key=jax.random.key(3))

# batch, heads, query_len, key_len
mask = jnp.ones((b, h, s, s), dtype=bool)
mask = mask.at[:, :, :, 10:].set(False)

attn_output_masked = jax.nn.dot_product_attention(q, k, v, mask = mask)
print(attn_output_masked.shape)
print(attn_output_masked)
mask2 = jnp.ones((b, h, s, 10), dtype=bool)
q = q[:, :10, :, :]
k = k[:, :10, :, :]
v = v[:, :10, :, :]
attn_output_unmasked = jax.nn.dot_product_attention(q, k, v)
print(attn_output_unmasked.shape)
print(attn_output_unmasked)
print(jnp.allclose(attn_output_masked[:, :10, :, :], attn_output_unmasked))