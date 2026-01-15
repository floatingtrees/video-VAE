import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Array
from einops import rearrange


class ConvBlock(nnx.Module):
    """Basic convolution block with norm and activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding='SAME',
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.norm = nnx.GroupNorm(num_groups=min(8, out_channels), num_features=out_channels,
                                   dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: Float[Array, "b h w c"]) -> Float[Array, "b h w c"]:
        x = self.conv(x)
        x = self.norm(x)
        x = nnx.silu(x)
        return x


class DownBlock(nnx.Module):
    """Downsampling block with two conv layers and max pooling."""
    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, rngs=rngs,
                               dtype=dtype, param_dtype=param_dtype)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3, rngs=rngs,
                               dtype=dtype, param_dtype=param_dtype)

    def __call__(self, x: Float[Array, "b h w c"]) -> tuple:
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        # 2x2 max pooling for downsampling
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skip


class UpBlock(nnx.Module):
    """Upsampling block with transposed conv and skip connection."""
    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()
        # Transposed conv for upsampling
        self.upsample = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(2, 2),
            strides=(2, 2),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        # After concatenation with skip, we have out_channels * 2
        self.conv1 = ConvBlock(out_channels * 2, out_channels, kernel_size=3, rngs=rngs,
                               dtype=dtype, param_dtype=param_dtype)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3, rngs=rngs,
                               dtype=dtype, param_dtype=param_dtype)

    def __call__(self, x: Float[Array, "b h w c"], skip: Float[Array, "b h2 w2 c2"]) -> Float[Array, "b h2 w2 c2"]:
        x = self.upsample(x)
        # Concatenate with skip connection
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nnx.Module):
    """
    UNet model for removing patch edges from video VAE outputs.

    Takes input shape (b, t, height, width, channels) and outputs the same shape.
    Uses kernels large enough to span multiple patches for smooth blending.
    """
    def __init__(self, channels: int, base_features: int = 32, num_levels: int = 3,
                 out_features: int = 3, rngs: nnx.Rngs = None,
                 dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        """
        Args:
            channels: Number of input/output channels (e.g., 3 for RGB)
            base_features: Base number of features, doubles at each level
            num_levels: Number of encoder/decoder levels (downsampling/upsampling stages)
            out_features: Number of output channels
            rngs: Random number generators for initialization
            dtype: Computation dtype (default bfloat16)
            param_dtype: Parameter storage dtype (default float32)
        """
        super().__init__()
        self.num_levels = num_levels
        self.dtype = dtype
        self.patch_mixer = nnx.Conv(in_features=channels, out_features=channels,
                                    kernel_size=(7, 7), padding='SAME',
                                    dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        # Encoder path
        self.encoders = []
        in_ch = channels
        for i in range(num_levels):
            out_ch = base_features * (2 ** i)
            self.encoders.append(DownBlock(in_ch, out_ch, rngs=rngs,
                                           dtype=dtype, param_dtype=param_dtype))
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = base_features * (2 ** num_levels)
        self.bottleneck1 = ConvBlock(in_ch, bottleneck_ch, kernel_size=3, rngs=rngs,
                                     dtype=dtype, param_dtype=param_dtype)
        self.bottleneck2 = ConvBlock(bottleneck_ch, bottleneck_ch, kernel_size=3, rngs=rngs,
                                     dtype=dtype, param_dtype=param_dtype)

        # Decoder path (reverse order of encoder)
        self.decoders = []
        in_ch = bottleneck_ch
        for i in range(num_levels - 1, -1, -1):
            out_ch = base_features * (2 ** i)
            self.decoders.append(UpBlock(in_ch, out_ch, rngs=rngs,
                                         dtype=dtype, param_dtype=param_dtype))
            in_ch = out_ch

        # Final output conv (no activation, we want clean output)
        self.final_conv = nnx.Conv(
            in_features=base_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding='SAME',
            dtype=dtype,
            kernel_init = nnx.initializers.zeros,
            param_dtype=param_dtype,
            rngs=rngs
        )


    def process_frame(self, x: Float[Array, "b h w c"]) -> Float[Array, "b h w c"]:
        """Process a single frame through the UNet."""
        # Encoder - collect skip connections
        skips = []
        x = self.patch_mixer(x)
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

    def __call__(self, x: Float[Array, "b time height width channels"]) -> Float[Array, "b time height width channels"]:
        """
        Process video input through UNet.

        Args:
            x: Input tensor of shape (b, t, h, w, c)

        Returns:
            Output tensor of shape (b, t, h, w, c) with smoothed patch boundaries
        """
        b, t, h, w, c = x.shape

        # Flatten batch and time for processing
        x = rearrange(x, "b t h w c -> (b t) h w c")

        # Cast to computation dtype
        x = x.astype(self.dtype)

        # Process through UNet
        x = self.process_frame(x)

        # Reshape back to video format
        x = rearrange(x, "(b t) h w c -> b t h w c", b=b, t=t)

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

    # Create UNet with configurable depth and mixed precision
    unet = UNet(channels=channels, base_features=32, num_levels=3, rngs=rngs,
                dtype=jnp.bfloat16, param_dtype=jnp.float32)

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
