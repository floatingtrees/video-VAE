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
    """3D downsampling block with two conv layers and spatiotemporal pooling."""
    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs,
                 temporal_kernel: int = 3,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels, kernel_size=3, rngs=rngs,
                                 temporal_kernel=temporal_kernel, dtype=dtype, param_dtype=param_dtype)
        self.conv2 = ConvBlock3D(out_channels, out_channels, kernel_size=3, rngs=rngs,
                                 temporal_kernel=temporal_kernel, dtype=dtype, param_dtype=param_dtype)

    @nnx.remat
    def __call__(self, x: Float[Array, "b t h w c"]) -> tuple:
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        # Spatial-only pooling (preserve temporal dimension)
        x = nnx.max_pool(x, window_shape=(1, 2, 2), strides=(1, 2, 2))
        return x, skip


class UpBlock3D(nnx.Module):
    """3D upsampling block with transposed conv and skip connection."""
    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs,
                 temporal_kernel: int = 3,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        # Transposed conv for spatial upsampling (preserve temporal dimension)
        self.upsample = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(1, 2, 2),
            strides=(1, 2, 2),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        # After concatenation with skip, we have out_channels * 2
        self.conv1 = ConvBlock3D(out_channels * 2, out_channels, kernel_size=3, rngs=rngs,
                                 temporal_kernel=temporal_kernel, dtype=dtype, param_dtype=param_dtype)
        self.conv2 = ConvBlock3D(out_channels, out_channels, kernel_size=3, rngs=rngs,
                                 temporal_kernel=temporal_kernel, dtype=dtype, param_dtype=param_dtype)

    @nnx.remat
    def __call__(self, x: Float[Array, "b t h w c"], skip: Float[Array, "b t h2 w2 c2"]) -> Float[Array, "b t h2 w2 c2"]:
        x = self.upsample(x)
        # Concatenate with skip connection
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nnx.Module):
    """
    3D UNet model for video processing with temporal convolutions.

    Takes input shape (b, t, height, width, channels) and outputs the same shape.
    Uses 3D convolutions to capture spatiotemporal patterns across frames.
    """
    def __init__(self, channels: int, base_features: int = 32, num_levels: int = 3,
                 out_features: int = 3, rngs: nnx.Rngs = None, temporal_kernel: int = 3,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        """
        Args:
            channels: Number of input/output channels (e.g., 3 for RGB)
            base_features: Base number of features, doubles at each level
            num_levels: Number of encoder/decoder levels (downsampling/upsampling stages)
            out_features: Number of output channels
            rngs: Random number generators for initialization
            temporal_kernel: Kernel size for temporal dimension (default 3)
            dtype: Computation dtype (default bfloat16)
            param_dtype: Parameter storage dtype (default float32)
        """
        super().__init__()
        self.num_levels = num_levels
        self.dtype = dtype
        # 3D patch mixer with temporal awareness
        self.patch_mixer = nnx.Conv(in_features=channels, out_features=channels,
                                    kernel_size=(temporal_kernel, 7, 7), padding='SAME',
                                    dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        # Encoder path
        self.encoders = []
        in_ch = channels
        for i in range(num_levels):
            out_ch = base_features * (2 ** i)
            self.encoders.append(DownBlock3D(in_ch, out_ch, rngs=rngs,
                                             temporal_kernel=temporal_kernel,
                                             dtype=dtype, param_dtype=param_dtype))
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = base_features * (2 ** num_levels)
        self.bottleneck1 = ConvBlock3D(in_ch, bottleneck_ch, kernel_size=3, rngs=rngs,
                                       temporal_kernel=temporal_kernel,
                                       dtype=dtype, param_dtype=param_dtype)
        self.bottleneck2 = ConvBlock3D(bottleneck_ch, bottleneck_ch, kernel_size=3, rngs=rngs,
                                       temporal_kernel=temporal_kernel,
                                       dtype=dtype, param_dtype=param_dtype)

        # Decoder path (reverse order of encoder)
        self.decoders = []
        in_ch = bottleneck_ch
        for i in range(num_levels - 1, -1, -1):
            out_ch = base_features * (2 ** i)
            self.decoders.append(UpBlock3D(in_ch, out_ch, rngs=rngs,
                                           temporal_kernel=temporal_kernel,
                                           dtype=dtype, param_dtype=param_dtype))
            in_ch = out_ch

        # Final output conv (no activation, we want clean output)
        self.final_conv = nnx.Conv(
            in_features=base_features,
            out_features=out_features,
            kernel_size=(1, 1, 1),
            padding='SAME',
            dtype=dtype,
            kernel_init=nnx.initializers.zeros,
            param_dtype=param_dtype,
            rngs=rngs
        )

    def __call__(self, x: Float[Array, "b time height width channels"]) -> Float[Array, "b time height width channels"]:
        """
        Process video input through 3D UNet.

        Args:
            x: Input tensor of shape (b, t, h, w, c)

        Returns:
            Output tensor of shape (b, t, h, w, c) with temporal coherence
        """
        # Cast to computation dtype
        x = x.astype(self.dtype)

        # Initial mixing with temporal context
        x = self.patch_mixer(x)

        # Encoder - collect skip connections
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        # Decoder with skip connections (reverse order)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Final projection
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    # Test the UNet
    seed = 42
    key = jax.random.key(seed)
    rngs = nnx.Rngs(seed)

    # Create test input
    batch_size = 2
    temporal_length = 8
    height = 256
    width = 256
    channels = 128

    input_video = jax.random.normal(key, (batch_size, temporal_length, height, width, channels)) * 0.02

    # Create 3D UNet with configurable depth and mixed precision
    unet = UNet(channels=channels, base_features=32, num_levels=3, rngs=rngs,
                temporal_kernel=3, dtype=jnp.bfloat16, param_dtype=jnp.float32)

    # JIT compile
    jit_forward = nnx.jit(unet.__call__)

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
    params = nnx.state(unet, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Trainable Parameters: {num_params / 10**6:.2f} Million")
