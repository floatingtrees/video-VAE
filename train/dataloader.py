import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Iterator, Optional, List, Tuple
import cv2
from functools import partial
from flax.jax_utils import prefetch_to_device


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


def load_video(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Load a video file and return as numpy array.
    
    Args:
        path: Path to the video file
        max_frames: Maximum number of frames to load (None for all)
    
    Returns:
        Video array in shape (T, H, W, C) with values in [0, 1]
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
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames loaded from video: {path}")
    
    # Stack frames: (T, H, W, C)
    video = np.stack(frames, axis=0)
    
    # Normalize to [0, 1]
    video = video.astype(np.float32) / 255.0
    
    return video


def load_video_as_jax(path: str, max_frames: Optional[int] = None) -> jnp.ndarray:
    """
    Load a video file and return as JAX array.
    
    Args:
        path: Path to the video file
        max_frames: Maximum number of frames to load (None for all)
    
    Returns:
        JAX array in shape (T, H, W, C) with values in [0, 1]
    """
    video_np = load_video(path, max_frames)
    return jnp.array(video_np)


class VideoDataset:
    """
    JAX-compatible video dataset that loads videos from /mnt/t9/videos/videos{i}.
    
    Videos are returned in standard format: (T, H, W, C) with values in [0, 1].
    """
    
    def __init__(
        self,
        base_dir: str = "/mnt/t9/videos",
        max_frames: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            base_dir: Base directory containing videos{i} subdirectories
            max_frames: Maximum number of frames to load per video
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
        """
        self.base_dir = base_dir
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.seed = seed
        
        self.video_paths = list_video_files(base_dir)
        
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.video_paths)
        
        print(f"Found {len(self.video_paths)} videos")
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> jnp.ndarray:
        """
        Load and return a single video as a JAX array.
        
        Returns:
            JAX array in shape (T, H, W, C) with values in [0, 1]
        """
        path = self.video_paths[idx]
        return load_video_as_jax(path, self.max_frames)
    
    def get_path(self, idx: int) -> str:
        """Get the file path for a given index."""
        return self.video_paths[idx]


class VideoDataLoader:
    """
    Batched dataloader for videos.
    
    Note: Since videos may have different lengths/resolutions,
    batching requires either:
    - Fixed-size videos (use resize and max_frames)
    - Batch size of 1
    - Custom padding/collation
    """
    
    def __init__(
        self,
        dataset: VideoDataset,
        batch_size: int = 1,
        resize: Optional[Tuple[int, int]] = None,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: VideoDataset instance
            batch_size: Number of videos per batch
            resize: Optional (H, W) to resize all videos
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.resize = resize
        self.drop_last = drop_last
    
    def _load_and_preprocess(self, idx: int) -> np.ndarray:
        """Load video and apply preprocessing."""
        path = self.dataset.video_paths[idx]
        video = load_video(path, self.dataset.max_frames)
        
        if self.resize is not None:
            h, w = self.resize
            t = video.shape[0]
            resized_frames = []
            for i in range(t):
                frame = cv2.resize(video[i], (w, h))
                resized_frames.append(frame)
            video = np.stack(resized_frames, axis=0)
        
        return video
    
    def __iter__(self) -> Iterator[jnp.ndarray]:
        """
        Iterate over batches.
        
        Yields:
            JAX arrays of shape (B, T, H, W, C) if batch_size > 1 and 
            videos have same dimensions, otherwise (T, H, W, C) for batch_size=1
        """
        num_samples = len(self.dataset)
        indices = list(range(num_samples))
        
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            
            if self.drop_last and (end_idx - start_idx) < self.batch_size:
                break
            
            batch_indices = indices[start_idx:end_idx]
            
            if self.batch_size == 1:
                # Single video, no batching needed
                yield self.dataset[batch_indices[0]]
            else:
                # Load and stack videos
                videos = [self._load_and_preprocess(idx) for idx in batch_indices]
                
                # For batching, all videos must have same shape
                # Pad temporal dimension to max length in batch
                max_t = max(v.shape[0] for v in videos)
                h, w, c = videos[0].shape[1:]
                
                padded_videos = []
                for v in videos:
                    if v.shape[0] < max_t:
                        pad_size = max_t - v.shape[0]
                        padding = np.zeros((pad_size, h, w, c), dtype=np.float32)
                        v = np.concatenate([v, padding], axis=0)
                    padded_videos.append(v)
                
                batch = np.stack(padded_videos, axis=0)
                yield jnp.array(batch)
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_dataloader(
    base_dir: str = "/mnt/t9/videos",
    batch_size: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = False,
) -> VideoDataLoader:
    """
    Convenience function to create a video dataloader.
    
    Args:
        base_dir: Base directory containing videos{i} subdirectories
        batch_size: Number of videos per batch
        max_frames: Maximum frames to load per video
        resize: Optional (H, W) to resize videos
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        drop_last: Whether to drop incomplete batches
    
    Returns:
        VideoDataLoader instance
    """
    dataset = VideoDataset(
        base_dir=base_dir,
        max_frames=max_frames,
        shuffle=shuffle,
        seed=seed,
    )
    
    return VideoDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        resize=resize,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    # Example usage
    naive_dataloader = create_dataloader(
        batch_size=1,
        max_frames=32,
        resize=(256, 256),
        shuffle=True,
    )
    
    print(f"Number of batches: {len(naive_dataloader)}")
    
    for i, batch in enumerate(naive_dataloader):
        print(f"Batch {i}: shape={batch.shape}, dtype={batch.dtype}, "
              f"min={batch.min():.3f}, max={batch.max():.3f}")
        if i >= 2:
            break
    prefetch_dataloader = prefetch_to_device(iter(naive_dataloader), size=2)