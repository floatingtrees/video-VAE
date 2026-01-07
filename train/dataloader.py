import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, List, Tuple, Dict
import cv2
import grain.python as grain


def list_video_files(base_dir: str = "/mnt/t9/videos") -> List[str]:
    """
    Collect all video files from /mnt/t9/videos/videos{i} directories.
    """
    video_paths = []
    
    # Find all videos{i} directories
    i = 0
    while True:
        dir_path = os.path.join(base_dir, f"videos{i}")
        if not os.path.isdir(dir_path):
            break
        
        # List all video files in this directory
        for filename in os.listdir(dir_path):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                video_paths.append(os.path.join(dir_path, filename))
        i += 1
    
    return video_paths


def load_video(
    path: str, 
    max_frames: Optional[int] = None, 
    resize: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a video file and return as numpy array with padding mask.
    
    Args:
        path: Path to the video file
        max_frames: Maximum number of frames to load (None for all). 
                    If video is shorter, it will be padded with zeros.
        resize: Optional (H, W) to resize frames
    
    Returns:
        Tuple of:
            - video: array in shape (T, H, W, C) with values in [0, 1]
            - mask: boolean array in shape (T,) with 1 for real frames, 0 for padded
    """
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    
    frames = []
    frame_count = 0
    
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
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
    
    def __init__(self, max_frames: Optional[int] = None, resize: Optional[Tuple[int, int]] = None):
        self.max_frames = max_frames
        self.resize = resize
    
    def map(self, path: str) -> Dict[str, np.ndarray]:
        """Load video from path and return as dict with video and mask."""
        video, mask = load_video(path, self.max_frames, self.resize)
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
        resize: Optional (H, W) to resize videos
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
        LoadVideoTransform(max_frames=max_frames, resize=resize),
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
        resize: Optional (H, W) to resize videos (required for batching)
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
        shard_options=grain.NoSharding(),
    )
    
    # Define transformations
    transformations = [
        LoadVideoTransform(max_frames=max_frames, resize=resize),
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
    # Example usage with Grain dataloader
    dataloader = create_dataloader(
        batch_size=1,
        max_frames=32,
        resize=(256, 256),
        shuffle=True,
        num_workers=4,
        prefetch_size=2,
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
        if i >= 2:
            break
    
    print("\nTesting batched Grain dataloader...")
    batched_dataloader = create_batched_dataloader(
        batch_size=4,
        max_frames=32,
        resize=(256, 256),
        shuffle=True,
        num_workers=4,
        prefetch_size=16,
        drop_remainder=True,
    )
    
    for i, batch in enumerate(batched_dataloader):
        video = jax.device_put(batch["video"], device)
        mask = jax.device_put(batch["mask"], device)
        # Sum real frames per video in batch
        real_frames_per_video = mask.sum(axis=1)
        print(f"Batch {i}: video shape={video.shape}, mask shape={mask.shape}, "
              f"real_frames_per_video={real_frames_per_video.tolist()}, "
              f"min={video.min():.3f}, max={video.max():.3f}")
        if i >= 2:
            break
