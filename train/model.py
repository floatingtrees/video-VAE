import jaxtyping
from beartype import beartype
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

class VideoEncoder(nn.Module):
    def __init__(self, num_frames: int, num_channels: int, num_height: int, num_width: int):
        super().__init__()
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.num_height = num_height
        self.num_width = num_width