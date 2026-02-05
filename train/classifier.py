import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Array


class ConvBlock3D(nnx.Module):
    """3D convolution block with norm and activation for spatiotemporal processing."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, rngs: nnx.Rngs,
                 temporal_kernel: int = 3,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(temporal_kernel, kernel_size, kernel_size),
            padding='SAME',
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
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

    # JIT compile
    jit_forward = nnx.jit(classifier.__call__)

    # Test forward pass
    import time
    start = time.perf_counter()
    output = jit_forward(input_video)
    print(f"First call (with compilation): {time.perf_counter() - start:.3f}s")
    print(f"Input shape: {input_video.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        output = jit_forward(input_video)
    print(f"10 iterations: {time.perf_counter() - start:.3f}s")

    # Count parameters
    params = nnx.state(classifier, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Trainable Parameters: {num_params / 10**6:.2f} Million")
