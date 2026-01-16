import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from einops import rearrange, repeat

class PatchEmbedding(nnx.Module):
    def __init__(self, height, width, channels, patch_size, rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.patch_size = patch_size
        self.dtype = dtype

        self.linear = nnx.Linear(patch_size * patch_size * channels, patch_size * patch_size * channels,
                                 dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.norm = nnx.LayerNorm(patch_size * patch_size * channels,
                                  dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: Float[Array, "b time height width channels"]):
        x = rearrange(x, "b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)",
            p1 = self.patch_size, p2 = self.patch_size)
        x = x.astype(self.dtype)
        x = self.norm(x)
        x = self.linear(x)

        return x

class PatchUnEmbedding(nnx.Module):
    def __init__(self, height, width, channels, patch_size, upsample_rate, rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.patch_size = patch_size
        self.height = height
        self.width = width
        self.upsample_rate = upsample_rate
        # We want more features for the unet to operate on
        self.upsample = nnx.Linear(patch_size * patch_size * channels, patch_size * patch_size * channels * upsample_rate,
                                   dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.downsample = nnx.Linear(channels * upsample_rate, channels,
                                     dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.linear = nnx.Linear(patch_size * patch_size * channels, patch_size * patch_size * channels,
                                 dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: Float[Array, "b time hw ppc"]):
        x = self.linear(x)
        x = self.upsample(x)
        convolutional_upsampled_features = rearrange(x, " b t (h w) (p1 p2 c u) -> b t (h p1) (w p2) (c u)",
            p1 = self.patch_size, p2 = self.patch_size,
            h = self.height // self.patch_size, w = self.width // self.patch_size,
            u = self.upsample_rate)
        x = self.downsample(convolutional_upsampled_features)


        return convolutional_upsampled_features, x

def create_sinusoidal_embeddings(
    seq_len: int,
    embed_dim: int,
    max_timescale: float = 10000.0
) -> Array:
    position = jnp.arange(seq_len, dtype=jnp.float32)[:, jnp.newaxis]

    div_term = jnp.exp(
        jnp.arange(0, embed_dim, 2, dtype=jnp.float32) * -(jnp.log(max_timescale) / embed_dim)
    )

    scaled_time = position * div_term[jnp.newaxis, :]

    sin_part = jnp.sin(scaled_time)
    cos_part = jnp.cos(scaled_time)

    pe = jnp.stack([sin_part, cos_part], axis=-1)

    pe = pe.reshape(seq_len, embed_dim)

    return pe[jnp.newaxis, :, :]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)

class RotaryEmbedding(nnx.Module):
    def __init__(self, head_dim, max_len=8192, alpha=1.0, base=10000.0):
        self.head_dim = head_dim
        self.max_len = max_len

        ntk_base = base * (alpha ** (head_dim / (head_dim - 2)))

        inv_freq = 1.0 / (ntk_base ** (jnp.arange(0, head_dim, 2) / head_dim))

        t = jnp.arange(max_len, dtype=jnp.float32)

        freqs = jnp.einsum("i,j->ij", t, inv_freq)


        emb = jnp.concatenate((freqs, freqs), axis=-1)

        self.cos_cached = nnx.Variable(jnp.cos(emb)[None, None, :, :])  # [1, 1, max_len, head_dim]
        self.sin_cached = nnx.Variable(jnp.sin(emb)[None, None, :, :])  # [1, 1, max_len, head_dim]

    def rotate_queries_and_keys(self, q, k):
        """
        Args:
            q, k: [Batch, Num_Heads, Seq_Len, Head_Dim]
        """
        seq_len = q.shape[2]

        cos_slice = jax.lax.dynamic_slice(
            self.cos_cached.value,
            (0, 0, 0, 0),
            (1, 1, seq_len, self.head_dim)
        )

        sin_slice = jax.lax.dynamic_slice(
            self.sin_cached.value,
            (0, 0, 0, 0),
            (1, 1, seq_len, self.head_dim)
        )

        # Apply rotation - cast to same dtype as q/k
        cos_slice = cos_slice.astype(q.dtype)
        sin_slice = sin_slice.astype(q.dtype)
        q_rot = (q * cos_slice) + (rotate_half(q) * sin_slice)
        k_rot = (k * cos_slice) + (rotate_half(k) * sin_slice)

        return q_rot, k_rot

class Attention(nnx.Module):
    def __init__(self, in_features, num_heads, qkv_features, max_len, use_qk_norm, rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        # use_qk_norm is legacy, now always True
        super().__init__()
        self.num_heads = num_heads
        head_dim = qkv_features // num_heads
        self.qkv_projection = nnx.Linear(in_features, qkv_features * 3,
                                         dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.out_projection = nnx.Linear(
            qkv_features,
            in_features,
            kernel_init=nnx.initializers.variance_scaling(
                scale=1e-2,  # 100x smaller than scale=1.0
                mode="fan_in",
                distribution="truncated_normal",
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.input_norm = nnx.LayerNorm(in_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.ROPE = RotaryEmbedding(head_dim = head_dim, max_len = max_len)
        self.use_qk_norm = use_qk_norm
        self.q_norm = nnx.LayerNorm(head_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs, use_bias=False)
        self.k_norm = nnx.LayerNorm(head_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs, use_bias=False)

    def __call__(self, x: Float[Array, "a seq dim"], mask: Float[Array, "b time 1 1 "] = None): # a = hw * b or b * t
        x = self.input_norm(x)
        q, k, v = jnp.split(self.qkv_projection(x), 3, axis = -1)
        q = rearrange(q, "b seq (head dim) -> b head seq dim", head = self.num_heads)
        k = rearrange(k, "b seq (head dim) -> b head seq dim", head = self.num_heads)
        v = rearrange(v, "b seq (head dim) -> b head seq dim", head = self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.ROPE.rotate_queries_and_keys(q, k)

        attn_output = jax.nn.dot_product_attention(q, k, v, mask = mask)
        attn_output = rearrange(attn_output, "b head seq dim -> b seq (head dim)")
        attn_output = self.out_projection(attn_output)
        return attn_output


class MLP(nnx.Module):
    def __init__(self, in_features, mlp_dim, rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.norm = nnx.LayerNorm(in_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.linear1 = nnx.Linear(in_features, mlp_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        self.linear2 = nnx.Linear(mlp_dim, in_features,
            kernel_init=nnx.initializers.variance_scaling(
                scale=1e-2,  # 100x smaller than scale=1.0
                mode="fan_in",
                distribution="truncated_normal",
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs)

    def __call__(self, x: Float[Array, "b time hw channels"]):
        x = self.norm(x)
        x = self.linear1(x)
        x = nnx.silu(x)
        x = self.linear2(x)
        return x

class FactoredAttention(nnx.Module):
    def __init__(self, mlp_dim, in_features, num_heads, qkv_features, max_temporal_len, max_spatial_len, rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.SpatialAttention = Attention(in_features, num_heads, qkv_features, max_spatial_len, True, rngs,
                                          dtype=dtype, param_dtype=param_dtype)
        self.SpatialMLP = MLP(in_features, mlp_dim, rngs, dtype=dtype, param_dtype=param_dtype)
        self.TemporalAttention = Attention(in_features, num_heads, qkv_features, max_temporal_len, False, rngs,
                                           dtype=dtype, param_dtype=param_dtype)
        self.TemporalMLP = MLP(in_features, mlp_dim, rngs, dtype=dtype, param_dtype=param_dtype)

    @nnx.remat
    def __call__(self, x: Float[Array, "b time hw channels"], temporal_mask: Float[Array, "b time 1 1 "]):
        b, t, hw, c = x.shape

        temporal_x = rearrange(x, "b t hw c -> (b hw) t c")
        temporal_attn_output = self.TemporalAttention(temporal_x, mask = temporal_mask)
        temporal_x = temporal_x + temporal_attn_output
        temporal_x = temporal_x + self.TemporalMLP(temporal_x)
        original_shape_x = rearrange(temporal_x, "(b hw) t c -> b t hw c", b = b, hw = hw)

        spatial_x = rearrange(original_shape_x, "b t hw c -> (b t) hw c")
        spatial_attn_output = self.SpatialAttention(spatial_x) # Only need mask to remove temporal frames
        spatial_x = spatial_x + spatial_attn_output
        spatial_x = spatial_x + self.SpatialMLP(spatial_x)

        original_shape_x = rearrange(spatial_x, "(b t) hw c -> b t hw c", b = b, t = t)
        return original_shape_x

@jax.custom_vjp
def round_ste(logits: Array) -> Array:
    return jnp.round(logits)

def round_ste_fwd(logits):
    return jnp.round(logits), ()

def round_ste_bwd(residuals, grad_output):
    return (grad_output, )

round_ste.defvjp(round_ste_fwd, round_ste_bwd)

class GumbelSigmoidSTE(nnx.Module):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, logits: Array, rngs: nnx.Rngs, train: bool = True) -> Array:
        if train:
            key = rngs.sampling()
            eps = 1e-20
            u = jax.random.uniform(key, logits.shape)
            u = jnp.clip(u, eps, 1.0 - eps)
            logistic_noise = jnp.log(u / (1-u))
            return round_ste(jax.nn.sigmoid((logits + logistic_noise) / self.temperature))
        else:
            # Deterministic: just threshold at 0.5 (sigmoid(0) = 0.5)
            return jnp.round(jax.nn.sigmoid(logits / self.temperature))
