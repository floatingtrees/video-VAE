import torch
from torch import nn 
from einops import rearrange

class VideoEncoder(nn.Module):
    def __init__(self, input_length, height, width): # Inputs in shape of (batch, num_frames, channels, height, width)
        super(VideoEncoder, self).__init__()
        self.PE = nn.Parameter(torch.randn(1, input_length, 1, input_channels)) 
        
    def forward(self, x): # 