import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from einops import rearrange, repeat

class PatchEmbedding(nnx.Module):
    def __init__(self, height, width, channels, patch_size, rngs: nnx.Rngs):
        super().__init__()
        self.patch_size = patch_size
        
        self.linear = nnx.Linear(patch_size * patch_size * channels, patch_size * patch_size * channels, rngs = rngs)
        self.norm = nnx.LayerNorm(patch_size * patch_size * channels, rngs = rngs)
        
    def __call__(self, x: Float[Array, "b time height width channels"]):
        x = rearrange(x, "b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)",
            p1 = self.patch_size, p2 = self.patch_size)
        x = self.norm(x)
        x = self.linear(x)
        
        return x

class PatchUnEmbedding(nnx.Module):
    def __init__(self, height, width, channels, patch_size, rngs: nnx.Rngs):
        super().__init__()
        self.patch_size = patch_size
        self.height = height
        self.width = width
        
        self.linear = nnx.Linear(patch_size * patch_size * channels, patch_size * patch_size * channels, rngs = rngs)
        
    def __call__(self, x: Float[Array, "b time hw ppc"]):
        x = self.linear(x)
        x = rearrange(x, " b t (h w) (p1 p2 c) -> b t (h p1) (w p2) c",
            p1 = self.patch_size, p2 = self.patch_size, 
            h = self.height // self.patch_size, w = self.width // self.patch_size)
        
        
        
        return x

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

class SinusoidalPositionalEncoding(nnx.Module):
    def __init__(self, max_len: int, embed_dim: int):
        self.pe_table = create_sinusoidal_embeddings(max_len, embed_dim)

    def __call__(self, x: Float[Array, "batch seq dim"]) -> Float[Array, "batch seq dim"]:
        seq_len = x.shape[1]
        return x + self.pe_table[:, :seq_len, :]

class Attention(nnx.Module):
    def __init__(self, in_features, num_heads, qkv_features, max_len, use_qk_norm, rngs: nnx.Rngs):
        super().__init__()
        self.MHA = nnx.MultiHeadAttention(num_heads = num_heads, 
            in_features = in_features, qkv_features = qkv_features, 
            rngs = rngs, decode = False, normalize_qk = use_qk_norm)
        self.PE = SinusoidalPositionalEncoding(max_len = max_len, embed_dim = in_features)
        
    def __call__(self, x: Float[Array, "a seq dim"], mask: Float[Array, "b 1 1 time"] = None):
        x = self.PE(x)
        attn_output = self.MHA(x, mask = mask)
        return attn_output

class MLP(nnx.Module):
    def __init__(self, in_features, mlp_dim, rngs: nnx.Rngs):
        super().__init__()
        self.norm = nnx.LayerNorm(in_features, rngs = rngs)
        self.linear1 = nnx.Linear(in_features, mlp_dim, rngs = rngs)
        
        self.linear2 = nnx.Linear(mlp_dim, in_features, rngs = rngs)
        
    def __call__(self, x: Float[Array, "b time hw channels"]):
        x = self.norm(x)
        x = self.linear1(x)
        x = nnx.silu(x)
        x = self.linear2(x)
        return x

class FactoredAttention(nnx.Module):
    def __init__(self, mlp_dim, in_features, num_heads, qkv_features, max_temporal_len, max_spatial_len, rngs: nnx.Rngs):
        super().__init__()
        self.SpatialAttention = Attention(in_features, num_heads, qkv_features, max_spatial_len, True, rngs)
        self.SpatialMLP = MLP(in_features, mlp_dim, rngs)
        self.TemporalAttention = Attention(in_features, num_heads, qkv_features, max_temporal_len, False, rngs)
        self.TemporalMLP = MLP(in_features, mlp_dim, rngs)
        self.norm = nnx.LayerNorm(in_features, rngs = rngs)

    def __call__(self, x: Float[Array, "b time hw channels"], temporal_mask: Float[Array, "b 1 1 time"]):
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
        return self.norm(original_shape_x)

@jax.custom_vjp 
def round_ste(logits: Array) -> Array:
    return jnp.round(logits)

def round_ste_fwd(logits):
    return jnp.round(logits), ()

def round_ste_bwd(residuals, grad_output):
    return grad_output

round_ste.defvjp(round_ste_fwd, round_ste_bwd)

class GumbelSigmoidSTE(nnx.Module):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, logits: Array, rngs: nnx.Rngs) -> Array:
        key = rngs.sampling()
        eps = 1e-20
        u = jax.random.uniform(key, logits.shape)
        u = jnp.clip(u, eps, 1.0 - eps)
        logistic_noise = jnp.log(u / 1-u)

        return round_ste(jax.nn.sigmoid((logits + logistic_noise) / self.temperature))

