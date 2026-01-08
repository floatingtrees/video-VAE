"""
Isolated attention masking test.
Tests that masking out frames 10+ produces the same output for frames 0-9
as running attention with only 10 frames.
"""
import jax
import jax.numpy as jnp
from flax import nnx
from layers import Attention

# Test parameters
seed = 42
key = jax.random.key(seed)
batch_size = 512
seq_len_full = 32
seq_len_cut = 10
dim = 64
num_heads = 4
qkv_features = 32
max_len = 64

print("=" * 60)
print("ATTENTION MASKING TEST")
print("=" * 60)

# Create attention module
attention = Attention(
    in_features=dim,
    num_heads=num_heads,
    qkv_features=qkv_features,
    max_len=max_len,
    rngs=nnx.Rngs(0)
)

# Create input
x_full = jax.random.normal(key, (batch_size, seq_len_full, dim))
x_cut = x_full[:, :seq_len_cut, :]

print(f"Full input shape: {x_full.shape}")
print(f"Cut input shape: {x_cut.shape}")

# Test 1: Additive mask (-inf for masked positions)
print("\n--- Test 1: Additive Mask (0 = attend, -inf = mask) ---")
mask_additive = jnp.zeros((batch_size, 1, 1, seq_len_full))
mask_additive = mask_additive.at[:, :, :, seq_len_cut:].set(float('-inf'))
print(f"Mask shape: {mask_additive.shape}")
print(f"Mask values [0,:,:,:5]: {mask_additive[0, 0, 0, :5]}")
print(f"Mask values [0,:,:,8:12]: {mask_additive[0, 0, 0, 8:12]}")

out_full_additive = attention(x_full, mask=mask_additive)
out_cut_no_mask = attention(x_cut, mask=None)

diff_additive = jnp.abs(out_full_additive[:, :seq_len_cut, :] - out_cut_no_mask)
print(f"Max diff (additive mask): {jnp.max(diff_additive):.6e}")
print(f"Mean diff (additive mask): {jnp.mean(diff_additive):.6e}")
print(f"Allclose (additive mask): {jnp.allclose(out_full_additive[:, :seq_len_cut, :], out_cut_no_mask)}")

# Test 2: Boolean mask (True = attend, False = mask)
print("\n--- Test 2: Boolean Mask (True = attend, False = mask) ---")
mask_bool = jnp.ones((batch_size, 1, 1, seq_len_full), dtype=bool)
mask_bool = mask_bool.at[:, :, :, seq_len_cut:].set(False)
print(f"Mask shape: {mask_bool.shape}")

try:
    out_full_bool = attention(x_full, mask=mask_bool)
    diff_bool = jnp.abs(out_full_bool[:, :seq_len_cut, :] - out_cut_no_mask)
    print(f"Max diff (boolean mask): {jnp.max(diff_bool):.6e}")
    print(f"Mean diff (boolean mask): {jnp.mean(diff_bool):.6e}")
    print(f"Allclose (boolean mask): {jnp.allclose(out_full_bool[:, :seq_len_cut, :], out_cut_no_mask)}")
except Exception as e:
    print(f"Boolean mask failed with: {e}")
exit()
# Test 3: 2D mask (query x key)
print("\n--- Test 3: 2D Mask (query_len x key_len) ---")
mask_2d = jnp.zeros((batch_size, 1, seq_len_full, seq_len_full))
mask_2d = mask_2d.at[:, :, :, seq_len_cut:].set(float('-inf'))
print(f"Mask shape: {mask_2d.shape}")

try:
    out_full_2d = attention(x_full, mask=mask_2d)
    diff_2d = jnp.abs(out_full_2d[:, :seq_len_cut, :] - out_cut_no_mask)
    print(f"Max diff (2D mask): {jnp.max(diff_2d):.6e}")
    print(f"Mean diff (2D mask): {jnp.mean(diff_2d):.6e}")
    print(f"Allclose (2D mask): {jnp.allclose(out_full_2d[:, :seq_len_cut, :], out_cut_no_mask)}")
except Exception as e:
    print(f"2D mask failed with: {e}")

# Test 4: Check raw MultiHeadAttention directly
print("\n--- Test 4: Raw nnx.MultiHeadAttention ---")
mha = nnx.MultiHeadAttention(
    num_heads=num_heads,
    in_features=dim,
    qkv_features=qkv_features,
    rngs=nnx.Rngs(1),
    decode=False
)

# No positional encoding - pure attention test
mask_raw = jnp.zeros((batch_size, 1, 1, seq_len_full))
mask_raw = mask_raw.at[:, :, :, seq_len_cut:].set(float('-inf'))

out_mha_full = mha(x_full, mask=mask_raw)
out_mha_cut = mha(x_cut, mask=None)

diff_mha = jnp.abs(out_mha_full[:, :seq_len_cut, :] - out_mha_cut)
print(f"Max diff (raw MHA): {jnp.max(diff_mha):.6e}")
print(f"Mean diff (raw MHA): {jnp.mean(diff_mha):.6e}")
print(f"Allclose (raw MHA): {jnp.allclose(out_mha_full[:, :seq_len_cut, :], out_mha_cut)}")

# Test 5: Same as Test 4 but with cut mask
print("\n--- Test 5: Raw MHA with explicit cut mask ---")
mask_cut = jnp.zeros((batch_size, 1, 1, seq_len_cut))
out_mha_cut_with_mask = mha(x_cut, mask=mask_cut)

diff_mha_cut = jnp.abs(out_mha_cut - out_mha_cut_with_mask)
print(f"Max diff (cut with/without mask): {jnp.max(diff_mha_cut):.6e}")
print(f"Allclose: {jnp.allclose(out_mha_cut, out_mha_cut_with_mask)}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
