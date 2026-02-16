import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Array

import jax
import jax.numpy as jnp
from flax import nnx

class ManualSpectralNorm(nnx.Module):
    def __init__(self, layer: nnx.Module, rngs: nnx.Rngs, n_steps: int = 1):
        self.layer = layer
        self.n_steps = n_steps
        
        # We need to find the weight kernel inside the wrapped layer.
        # Most layers (Linear, Conv) call it 'kernel'.

        # Initialize the 'u' vector for power iteration
        # Shape is (1, out_features) usually, but we assume (1, W.shape[1])
        kernel_shape = self.layer.kernel.value.shape
        self.u = nnx.BatchStat(jax.random.normal(rngs.params(), (1, kernel_shape[-1])))

    def __call__(self, x, update_stats: bool = True):

        weight = self.layer.kernel.value
        weight_mat = weight.reshape(-1, weight.shape[-1])
        u = self.u.value
        v = None

        # Power iteration to approximate spectral norm
        # This is the standard algorithm used in Miyato et al. (2018)
        if update_stats:
            for _ in range(self.n_steps):
                # v = normalized(u @ W.T)
                v = jnp.dot(u, weight_mat.T)
                v = v / jnp.linalg.norm(v, keepdims=True)

                # u = normalized(v @ W)
                u = jnp.dot(v, weight_mat)
                u = u / jnp.linalg.norm(u, keepdims=True)

            # Update the state in-place
            self.u.value = u

        # Calculate sigma (spectral norm) using the latest u and v
        # We recompute v here to ensure consistency if update_stats was False
        if v is None:
             v = jnp.dot(u, weight_mat.T)
             v = v / jnp.linalg.norm(v, keepdims=True)

        # sigma = v @ W @ u.T
        weight_sn = jnp.dot(jnp.dot(v, weight_mat), u.T)
        sigma = weight_sn[0, 0]

        # Apply the normalization to the weight for this forward pass only
        # We hack the layer's kernel temporarily
        original_kernel = self.layer.kernel.value
        self.layer.kernel.value = original_kernel / sigma
        
        try:
            # Run the wrapped layer
            out = self.layer(x)
        finally:
            # Restore the original un-normalized weights so the optimizer 
            # updates the real weights, not the normalized ones.
            self.layer.kernel.value = original_kernel
            
        return out

class ConvBlock3D(nnx.Module):
    """3D convolution block with norm and activation for spatiotemporal processing."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, rngs: nnx.Rngs,
                 temporal_kernel: int = 3,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.conv = ManualSpectralNorm(nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(temporal_kernel, kernel_size, kernel_size),
            padding='SAME',
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        ), rngs=rngs)
        self.norm = nnx.GroupNorm(num_groups=min(8, out_channels), num_features=out_channels,
                                   dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: Float[Array, "b t h w c"]) -> Float[Array, "b t h w c"]:
        x = self.conv(x)
        x = self.norm(x)
        x = nnx.silu(x)
        return x


class DownBlock3D(nnx.Module):
    """3D downsampling block with two conv layers and spatial pooling."""
    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs,
                 temporal_kernel: int = 3,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels, kernel_size=3, rngs=rngs,
                                 temporal_kernel=temporal_kernel, dtype=dtype, param_dtype=param_dtype)
        self.conv2 = ConvBlock3D(out_channels, out_channels, kernel_size=3, rngs=rngs,
                                 temporal_kernel=temporal_kernel, dtype=dtype, param_dtype=param_dtype)

    @nnx.remat
    def __call__(self, x: Float[Array, "b t h w c"]) -> Float[Array, "b t h2 w2 c"]:
        x = self.conv1(x)
        x = self.conv2(x)
        # Spatial-only pooling (preserve temporal dimension)
        x = nnx.max_pool(x, window_shape=(1, 2, 2), strides=(1, 2, 2))
        return x


class Classifier(nnx.Module):
    """
    3D CNN classifier for video real/fake classification.

    Takes input shape (b, t, height, width, channels) and outputs (b,) logits.
    Uses 3D convolutions to capture spatiotemporal patterns across frames.
    No assumptions on the time dimension of the input.
    """
    def __init__(self, channels: int, base_features: int = 32, num_levels: int = 4,
                 rngs: nnx.Rngs = None, temporal_kernel: int = 3,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        """
        Args:
            channels: Number of input channels (e.g., 3 for RGB)
            base_features: Base number of features, doubles at each level
            num_levels: Number of encoder levels (downsampling stages)
            rngs: Random number generators for initialization
            temporal_kernel: Kernel size for temporal dimension (default 3)
            dtype: Computation dtype (default bfloat16)
            param_dtype: Parameter storage dtype (default float32)
        """
        super().__init__()
        self.num_levels = num_levels
        self.dtype = dtype

        # Initial 3D conv to expand channels
        self.initial_conv = ConvBlock3D(channels, base_features, kernel_size=7, rngs=rngs,
                                        temporal_kernel=temporal_kernel,
                                        dtype=dtype, param_dtype=param_dtype)

        # Encoder path
        self.encoders = []
        in_ch = base_features
        for i in range(num_levels):
            out_ch = base_features * (2 ** (i + 1))
            self.encoders.append(DownBlock3D(in_ch, out_ch, rngs=rngs,
                                             temporal_kernel=temporal_kernel,
                                             dtype=dtype, param_dtype=param_dtype))
            in_ch = out_ch

        # Final features dimension
        final_features = base_features * (2 ** num_levels)

        # Classification head
        self.classifier = nnx.Linear(final_features, 1, rngs=rngs,
                                     dtype=dtype, param_dtype=param_dtype)

    def __call__(self, x: Float[Array, "b time height width channels"], mask = None, train: bool = True) -> Float[Array, "b"]:
        """
        Classify video input as real or fake.

        Args:
            x: Input tensor of shape (b, t, h, w, c)

        Returns:
            Logits tensor of shape (b,) - single logit per sample
        """

        # Initial conv
        x = self.initial_conv(x)

        # Encoder - progressive downsampling
        for encoder in self.encoders:
            x = encoder(x)

        # Global average pooling over time, height, width
        x = jnp.mean(x, axis=(1, 2, 3))  # (b, c)

        # Classification
        x = self.classifier(x)  # (b, 1)

        return x



if __name__ == "__main__":
    # Test the Classifier
    seed = 42
    key = jax.random.key(seed)
    rngs = nnx.Rngs(seed)

    # Create test input
    batch_size = 2
    temporal_length = 8
    height = 256
    width = 256
    channels = 3

    input_video = jax.random.normal(key, (batch_size, temporal_length, height, width, channels)) * 0.02

    # Create 3D CNN Classifier
    classifier = Classifier(channels=channels, base_features=32, num_levels=4, rngs=rngs,
                            temporal_kernel=3, dtype=jnp.bfloat16, param_dtype=jnp.float32)
    #print(classifier.initial_conv.conv.__dict__);exit()

    # JIT compile
    @nnx.jit
    def jit_forward(model, x):
        return model(x)

    # Test forward pass
    import time
    start = time.perf_counter()
    output = jit_forward(classifier, input_video)
    print(f"First call (with compilation): {time.perf_counter() - start:.3f}s")
    print(f"Input shape: {input_video.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        output = jit_forward(classifier, input_video)
    print(f"10 iterations: {time.perf_counter() - start:.3f}s")

    # Count parameters
    params = nnx.state(classifier, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Trainable Parameters: {num_params / 10**6:.2f} Million")

    # Verify spectral norm
    sn_layer = classifier.initial_conv.conv  # ManualSpectralNorm
    kernel = sn_layer.layer.kernel.value
    kernel_mat = kernel.reshape(-1, kernel.shape[-1]).astype(jnp.float32)
    true_sigma = jnp.linalg.svd(kernel_mat, compute_uv=False)[0]
    u = sn_layer.u.value.astype(jnp.float32)
    v = jnp.dot(u, kernel_mat.T)
    v = v / jnp.linalg.norm(v, keepdims=True)
    estimated_sigma = jnp.dot(jnp.dot(v, kernel_mat), u.T)[0, 0]
    print(f"\nSpectral norm check (initial_conv):")
    print(f"  True spectral norm (SVD): {true_sigma:.6f}")
    print(f"  Estimated sigma (power iter): {estimated_sigma:.6f}")
