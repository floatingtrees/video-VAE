import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
import cv2
import grain.python as grain
from random import randint

def batch_to_video(
    batch: Dict[str, Union[np.ndarray, jnp.ndarray]],
    output_path: str,
    fps: float = 30.0,
    use_mask: bool = True,
    sample_idx: int = 0,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """
    Extract the first (or specified) sample from a batch and save it as a video file.
    
    Args:
        batch: Dict with 'video' and 'mask' keys from the dataloader.
               'video' shape: (T, H, W, C) or (B, T, H, W, C)
               'mask' shape: (T,) or (B, T)
        output_path: Path to save the output video (e.g., 'output.mp4')
        fps: Frames per second for the output video
        use_mask: If True, only include real frames (mask=1), excluding padded frames
        sample_idx: Index of the sample to extract from batch (default 0)
        crf: Constant rate factor for H.264 encoding (lower = better quality)
        preset: ffmpeg encoding preset (ultrafast, fast, medium, slow, etc.)
    """
    import subprocess
    import shutil
    
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    
    video = batch["video"]
    mask = batch["mask"]
    video = jnp.clip(video, 0, 1)
    
    # Convert JAX arrays to numpy
    if hasattr(video, "device"):
        video = np.array(video)
    if hasattr(mask, "device"):
        mask = np.array(mask)
    
    # Handle batched vs unbatched
    if video.ndim == 5:  # (B, T, H, W, C)
        video = video[sample_idx]
        mask = mask[sample_idx]
    
    # Convert from [0, 1] to [0, 255]
    video = (video * 255).astype(np.uint8)
    
    # Filter out padded frames if use_mask is True
    if use_mask:
        real_frames = mask > 0.5
        video = video[real_frames]
    
    if len(video) == 0:
        raise ValueError("No frames to write (all frames are padded)")
    
    # Get video dimensions
    t, h, w, c = video.shape
    
    # Pipe raw frames directly to ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
        output_path
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame in video:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg failed.")
    
    print(f"Saved video to {output_path} ({len(video)} frames, {w}x{h}, {fps} fps)")


def list_video_files(base_dir: str = "/mnt/t9/videos") -> List[str]:
    """
    Collect all video files from /mnt/t9/videos/videos{i} directories.
    """
    video_paths = []
    
    # Find all videos{i} directories
    for i in range(0, 100):
        dir_path = os.path.join(base_dir, f"videos{i}")
        if not os.path.isdir(dir_path):
            continue
        
        # List all video files in this directory
        for filename in os.listdir(dir_path):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                video_paths.append(os.path.join(dir_path, filename))
    
    return video_paths


def get_random_crop_params(h: int, w: int, crop_size: int) -> Tuple[int, int, int, int]:
    """
    Get random crop parameters. Returns (new_h, new_w, start_h, start_w).
    If the frame is smaller than crop_size, computes resize dimensions first.
    """
    # If frame is smaller than crop_size, compute resize dimensions
    if h < crop_size or w < crop_size:
        scale = max(crop_size / h, crop_size / w)
        h, w = int(h * scale), int(w * scale)
    
    # Random crop position
    start_h = np.random.randint(0, h - crop_size + 1)
    start_w = np.random.randint(0, w - crop_size + 1)
    
    return h, w, start_h, start_w


def apply_crop(frame: np.ndarray, crop_size: int, crop_params: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Apply crop to a frame using pre-computed crop parameters.
    """
    target_h, target_w, start_h, start_w = crop_params
    h, w = frame.shape[:2]
    
    # Resize if needed
    if h != target_h or w != target_w:
        frame = cv2.resize(frame, (target_w, target_h))
    
    # Apply crop
    return frame[start_h:start_h + crop_size, start_w:start_w + crop_size]


def load_video(
    path: str, 
    max_frames: Optional[int] = None, 
    resize: Optional[Tuple[int, int]] = None,
    crop_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a video file and return as numpy array with padding mask.
    
    Args:
        path: Path to the video file
        max_frames: Maximum number of frames to load (None for all). 
                    If video is shorter, it will be padded with zeros.
        resize: Optional (H, W) to resize frames after random cropping
        crop_size: Size of the random crop (default 512x512)
    
    Returns:
        Tuple of:
            - video: array in shape (T, H, W, C) with values in [0, 1]
            - mask: boolean array in shape (T,) with 1 for real frames, 0 for padded
    """
    try:
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        
        frames = []
        frame_count = 0
        crop_params = None  # Will be set on first frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = randint(0, max(total_frames - max_frames, 0))
        counter = 0
        while True:
            ret, frame = cap.read()
            if counter < start_frame:
                counter += 1
                continue
            if max_frames is not None and frame_count >= max_frames:
                break
                
            
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Compute random crop params on first frame (same crop for all frames)
            if crop_params is None:
                h, w = frame.shape[:2]
                crop_params = get_random_crop_params(h, w, crop_size)
            
            # Apply random crop (same position for all frames)
            frame = apply_crop(frame, crop_size, crop_params)
            
            # Resize if specified
            if resize is not None:
                h, w = resize
                frame = cv2.resize(frame, (w, h))
            
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {path}")
        
        # Stack frames: (T, H, W, C)
        video = np.stack(frames, axis=0)
        
        # Normalize to [0, 1]
        video = video.astype(np.float32) / 255.0
        
        num_real_frames = video.shape[0]
        
        # Pad to max_frames if video is shorter
        if max_frames is not None and video.shape[0] < max_frames:
            pad_size = max_frames - video.shape[0]
            padding = np.zeros((pad_size, *video.shape[1:]), dtype=np.float32)
            video = np.concatenate([video, padding], axis=0)
        
        # Create mask: 1 for real frames, 0 for padded
        total_frames = video.shape[0]
        mask = np.zeros(total_frames, dtype=np.float32)
        mask[:num_real_frames] = 1.0
    except Exception as e:
        print(e, path) 
        h, w = resize
        video = np.zeros((max_frames, h, w, 3), dtype = np.float32)
        mask = np.ones(max_frames, dtype = np.float32)
    return video, mask


class VideoDataSource(grain.RandomAccessDataSource):
    """
    Grain-compatible data source for video files.
    """
    
    def __init__(self, base_dir: str = "/mnt/t9/videos"):
        self.video_paths = list_video_files(base_dir)
        print(f"Found {len(self.video_paths)} videos")
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> str:
        """Return the video path at the given index."""
        return self.video_paths[idx]


class LoadVideoTransform(grain.MapTransform):
    """
    Grain transform to load and preprocess a video from path.
    Returns a dict with 'video' and 'mask' keys.
    """
    
    def __init__(self, max_frames: Optional[int] = None, resize: Optional[Tuple[int, int]] = None, crop_size: int = 512):
        self.max_frames = max_frames
        self.resize = resize
        self.crop_size = crop_size
    
    def map(self, path: str) -> Dict[str, np.ndarray]:
        """Load video from path and return as dict with video and mask."""
        video, mask = load_video(path, self.max_frames, self.resize, self.crop_size)
        return {"video": video, "mask": mask}


class ToJaxTransform(grain.MapTransform):
    """
    Grain transform to convert numpy arrays to JAX arrays on device.
    """
    
    def __init__(self, device: Optional[jax.Device] = None):
        self.device = device or jax.devices()[0]
    
    def map(self, data: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
        """Convert numpy arrays to JAX arrays on device."""
        return {
            "video": jax.device_put(jnp.array(data["video"]), self.device),
            "mask": jax.device_put(jnp.array(data["mask"]), self.device),
        }


def create_dataloader(
    base_dir: str = "/mnt/t9/videos",
    batch_size: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    crop_size: int = 512,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 4,
    prefetch_size: int = 2,
) -> grain.DataLoader:
    """
    Create a Grain-based video dataloader with prefetching.
    
    Args:
        base_dir: Base directory containing videos{i} subdirectories
        batch_size: Number of videos per batch
        max_frames: Maximum frames to load per video. Videos shorter than this
                    will be padded with zeros.
        resize: Optional (H, W) to resize videos after center cropping
        crop_size: Size of center crop before resize (default 512)
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        num_workers: Number of worker processes for data loading
        prefetch_size: Number of batches to prefetch
    
    Returns:
        Grain DataLoader instance that yields dicts with:
            - 'video': array of shape (T, H, W, C)
            - 'mask': array of shape (T,) with 1 for real frames, 0 for padded
    """
    # Create data source
    data_source = VideoDataSource(base_dir)
    
    # Create index sampler
    if shuffle:
        sampler = grain.IndexSampler(
            num_records=len(data_source),
            shuffle=True,
            seed=seed,
            shard_options=grain.NoSharding(),
        )
    else:
        sampler = grain.IndexSampler(
            num_records=len(data_source),
            shuffle=False,
            shard_options=grain.NoSharding(),
        )
    
    # Define transformations
    transformations = [
        LoadVideoTransform(max_frames=max_frames, resize=resize, crop_size=crop_size),
    ]
    
    # Create dataloader
    dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=num_workers,
        read_options=grain.ReadOptions(prefetch_buffer_size=prefetch_size),
    )
    
    return dataloader


def create_batched_dataloader(
    base_dir: str = "/mnt/t9/videos",
    batch_size: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    crop_size: int = 512,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 4,
    prefetch_size: int = 16,
    drop_remainder: bool = False,
) -> grain.DataLoader:
    """
    Create a Grain-based video dataloader with batching and prefetching.
    
    Args:
        base_dir: Base directory containing videos{i} subdirectories
        batch_size: Number of videos per batch
        max_frames: Maximum frames to load per video (required for batching).
                    Videos shorter than this will be padded with zeros.
        resize: Optional (H, W) to resize videos after center cropping (required for batching)
        crop_size: Size of center crop before resize (default 512)
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        num_workers: Number of worker processes for data loading
        prefetch_size: Number of batches to prefetch
        drop_remainder: Whether to drop the last incomplete batch
    
    Returns:
        Grain DataLoader instance that yields dicts with:
            - 'video': array of shape (B, T, H, W, C)
            - 'mask': array of shape (B, T) with 1 for real frames, 0 for padded
    """
    # Create data source
    data_source = VideoDataSource(base_dir)
    
    # Create index sampler
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=1,
        shard_options=grain.NoSharding(),
    )
    
    # Define transformations
    transformations = [
        LoadVideoTransform(max_frames=max_frames, resize=resize, crop_size=crop_size),
        grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder),
    ]
    
    # Create dataloader
    dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=num_workers,
        read_options=grain.ReadOptions(prefetch_buffer_size=prefetch_size),
    )
    
    return dataloader


if __name__ == "__main__":
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Example usage with Grain dataloader
    dataloader = create_dataloader(
        batch_size=1,
        max_frames=1024,
        resize=(256, 256),
        shuffle=True,
        num_workers=0,
        prefetch_size=2,
        seed=10
    )
    
    print("Testing Grain dataloader...")
    device = jax.devices()[0]
    
    for i, batch in enumerate(dataloader):

        # Transfer to GPU
        video = jax.device_put(batch["video"], device)
        mask = jax.device_put(batch["mask"], device)
        num_real = int(mask.sum())
        print(f"Batch {i}: video shape={video.shape}, mask shape={mask.shape}, "
              f"real_frames={num_real}/{video.shape[0]}, "
              f"min={video.min():.3f}, max={video.max():.3f}")
        
        # Save first batch as video
        if i == 1:
            batch_to_video(batch, os.path.join(output_dir, "sample_unbatched.mp4"), fps=30.0)
        
        if i >= 200:
            break
    
    print("\nTesting batched Grain dataloader...")
    exit()
    batched_dataloader = create_batched_dataloader(
        batch_size=4,
        max_frames=128,
        resize=(256, 256),
        shuffle=True,
        num_workers=4,
        prefetch_size=16,
        drop_remainder=True,
    )
    
    for i, batch in enumerate(batched_dataloader):
        video = jax.device_put(batch["video"], device)
        mask = jax.device_put(batch["mask"], device)
        print(mask)
        print(mask[0][0])
        # Sum real frames per video in batch
        real_frames_per_video = mask.sum(axis=1)
        print(f"Batch {i}: video shape={video.shape}, mask shape={mask.shape}, "
              f"real_frames_per_video={real_frames_per_video.tolist()}, "
              f"min={video.min():.3f}, max={video.max():.3f}")
        
        # Save first sample from first batch as video
        if i == 1:
            batch_to_video(batch, os.path.join(output_dir, "sample_batched.mp4"), fps=30.0, sample_idx=0)
        
        if i >= 2:
            break
    print("DONE")
    import time 
    time.sleep(1000)