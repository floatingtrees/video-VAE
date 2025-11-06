import torch
from torch import nn 
from einops import rearrange, repeat, reduce
from rotary_embedding_torch import RotaryEmbedding


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim = dim)

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        attn_output, _ = self.attn(q, k, v)
        return attn_output

class Chunker(nn.Module):
    def __init__(self, channels, height, width, inner_dim, depth): # (batch, num_frames, height*width, channels)
        super(Chunker, self).__init__()
        self.linear1 = nn.Linear(channels * height * width, inner_dim)
        self.linear2 = nn.Linear(inner_dim, 1)
        self.SiLU = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(inner_dim)
        self.attn = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=1, batch_first=True)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(inner_dim),
                MLP(inner_dim)
            ]))
        
    def forward(self, x): # (batch, num_frames, height*width, channels)
        x = rearrange(x, 'b (x) c -> b (x c)')
        x = self.linear1(x)
        for attn, mlp in self.layers:
            x = x + attn(x)
            x = x + mlp(x)
        output = self.sigmoid(self.linear2(x))  # (batch, 1)
        return x, output
    
        
# LATENTS IN SHAPE OF (128, 8, 8)
class VideoEncoder(nn.Module):
    def __init__(self, channels, height, width, inner_dim=64, depth=2): # Inputs in shape of (batch, num_frames, channels, height, width)
        super(VideoEncoder, self).__init__()
        # This is the PE for the image patches, so it can be constant
        self.image_PE = nn.Parameter(torch.randn(height * width, channels)) * 0.02 # initialize with low std
        self.Chunker = Chunker(channels, height, width, inner_dim=inner_dim, depth=depth)
        
    def encode(x, masks, cutoffs):
        pass
    
    def forward(self, x, masks): # outputs in shape of (batch, compressed_dim, 1)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (batch, num_frames, height*width, channels)
        x = x + self.image_PE  # Add positional encoding
        chunks = self.Chunker(x)  # (batch, 1)
        
if __name__ == "__main__":
    model = VideoEncoder(channels=128, height=8, width=8)
    total_params = 0
    for element in model.parameters():
        total_params += element.numel()
    print("Total number of parameters: ", round(total_params / 10**6, 2), "Million")
    dummy_input = torch.randn(2, 128, 8, 8)  # (batch, channels, height, width)
    dummy_mask = None
    output = model(dummy_input, dummy_mask)