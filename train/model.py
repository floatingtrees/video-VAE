import torch
from torch import nn 
from einops import rearrange, repeat, reduce
from rotary_embedding_torch import RotaryEmbedding
from dataset import LatentDataset, collate_fn
from torch.utils.data import DataLoader
from torch import Tensor
import time


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
    def __init__(self, dim, num_heads = 4):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim = dim)

    def forward(self, x, mask):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        if mask is not None:
            mask = repeat(mask, "b s1 s2 -> (b h) s1 s2", h = self.num_heads)
        attn_output, _ = self.attn(q, k, v, attn_mask = mask)
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
        
    def forward(self, x, mask): # (batch, num_frames, height*width, channels)
        x = rearrange(x, 'b s (hw) c -> b s (hw c)')
        x = self.linear1(x)
        for attn, mlp in self.layers:
            x = x + attn(x, mask)
            x = x + mlp(x)
        output = self.sigmoid(self.linear2(x))  # (batch, 1)
        return x, output

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)

class Compressor(nn.Module):
    def __init__(self, channels, height, width, temporal_dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Projection(channels * height * width, temporal_dim),
                Attention(temporal_dim),
                Projection(temporal_dim, channels * height * width),
                Attention(channels),
                MLP(channels)
            ]))
    def forward(self, x, split_attn_mask):
        b, s, hw, c = x.shape
        for down_proj, temporal_attn, up_proj, spatial_attn, mlp in self.layers:
            temporal_reshaped = down_proj(rearrange(x, 'b s (hw) c -> b s (hw c)'))
            temporal_attn_output = up_proj(temporal_attn(temporal_reshaped, split_attn_mask))
            temporal_inverse_reshaped = rearrange(temporal_attn_output, "b s (hw c) -> b s (hw) c", hw = hw)
            x = x + temporal_inverse_reshaped
            spatial_reshaped = rearrange(x, 'b s (hw) c -> (b s) hw c')
            spatial_attn_output = spatial_attn(spatial_reshaped, mask = None)
            spatial_inverse_reshaped = rearrange(spatial_attn_output, "(b s) hw c -> b s (hw) c", b = b, s = s)
            x = x + spatial_inverse_reshaped
            x = x + mlp(x)
        return x
        
def sinusoidal_positions(L, D, device=None, dtype=None, base=10000.0, offset=0):
    """
    Returns [L, D] sinusoidal embeddings (even dims = sin, odd dims = cos).
    offset lets you resume for cached KV with past positions.
    """
    device = device or torch.device('cpu')
    dtype = dtype or torch.get_default_dtype()
    pos = torch.arange(offset, offset + L, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(0, D, 2, device=device, dtype=dtype).unsqueeze(0)
    inv_freq = (base ** (-i / D))
    angles = pos * inv_freq
    pe = torch.empty(L, D, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe

class Decompressor(nn.Module):
    def __init__(self, channels, height, width, temporal_dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Projection(channels * height * width, temporal_dim),
                Attention(temporal_dim),
                Projection(temporal_dim, channels * height * width),
                Attention(channels),
                MLP(channels)
            ]))
    def forward(self, x, split_attn_mask):
        b, s, hw, c = x.shape
        for down_proj, temporal_attn, up_proj, spatial_attn, mlp in self.layers:
            temporal_reshaped = down_proj(rearrange(x, 'b s (hw) c -> b s (hw c)'))
            temporal_attn_output = up_proj(temporal_attn(temporal_reshaped, split_attn_mask))
            temporal_inverse_reshaped = rearrange(temporal_attn_output, "b s (hw c) -> b s (hw) c", hw = hw)
            x = x + temporal_inverse_reshaped
            spatial_reshaped = rearrange(x, 'b s (hw) c -> (b s) hw c')
            spatial_attn_output = spatial_attn(spatial_reshaped, mask = None)
            spatial_inverse_reshaped = rearrange(spatial_attn_output, "(b s) hw c -> b s (hw) c", b = b, s = s)
            x = x + spatial_inverse_reshaped
            x = x + mlp(x)
        return x
        

# LATENTS IN SHAPE OF (128, 8, 8)
class VideoEncoder(nn.Module):
    def __init__(self, channels, height, width, temporal_dim = 128, inner_dim=64, depth=5): # Inputs in shape of (batch, num_frames, channels, height, width)
        super(VideoEncoder, self).__init__()
        # This is the PE for the image patches, so it can be constant
        self.image_PE = nn.Parameter(torch.zeros(height * width, channels))
        nn.init.normal_(self.image_PE, mean=0.0, std=0.02)
        self.chunker = Chunker(channels, height, width, inner_dim=inner_dim, depth=depth)
        self.compressor = Compressor(channels, height, width, temporal_dim=temporal_dim, depth=depth)
        self.decompressor = Decompressor(channels, height, width, temporal_dim=temporal_dim, depth=depth)
        
    def encode(x, masks, cutoffs):
        pass
    
    def forward(self, latents: Tensor, split_attn_mask: Tensor, 
                global_attention_mask: Tensor, compression_mask: Tensor, decompression_mask: Tensor
                ): 
        '''
        x: Tensor in shape of (batch, seq_length, channels, height, width), for latent representation
        split_attn_mask: Tensor in shape of (batch, seq_length, seq_legth) for masking out chunks 
        global_attention_mask: Tensor in shape of (batch, seq_length, seq_legth) for initial attention masking
        compression_mask: Tensor in shape of (batch, seq_length) that removes everything but the 1st and last frames of each chunk
        decompression_mask: Tensor in shape (batch, seq_length, seq_length) that broadcasts 1st frame 
        '''
        start = time.perf_counter()
        b, s, c, h, w  = latents.shape
        x = rearrange(latents, 'b s c h w -> b s (h w) c')  # (batch, num_frames, height*width, channels)
        x = x + self.image_PE  # Add positional encoding
        attn_scores, chunks = self.chunker(x, global_attention_mask)  # (batch, 1)
        compressed = self.compressor(x, split_attn_mask)
        print(time.perf_counter() - start)
        compression_mask = rearrange(compression_mask, "b s -> b s 1 1")
        compressed = compressed * compression_mask
        
        decompressed = rearrange(compressed, "b s (h w) c -> (h w) b c s", h = h, w = w)
        decompressed = torch.matmul(decompressed, decompression_mask)
        decompressed = rearrange(decompressed, "(h w) b c s -> b s (h w) c", h = h, w=w)
        #assert torch.allclose(decompressed[0, 0], decompressed[0, 10])
        #assert torch.allclose(decompressed, compressed)

        embedding_tensor = sinusoidal_positions(s, c * h * w, device = decompressed.device)
        embedding_tensor = rearrange(embedding_tensor, "s (c h w) -> s (h w) c", h=h, c=c, w=w)
        decompressed = decompressed + embedding_tensor
        reconstruction = self.decompressor(decompressed, split_attn_mask)
        return reconstruction, chunks
        
        
        
        
       

        
        
        
        
if __name__ == "__main__":
    device = "cuda"
    model = VideoEncoder(channels=128, height=8, width=8)
    model.to(device)
    total_params = 0
    for element in model.parameters():
        total_params += element.numel()
    print("Total number of parameters: ", round(total_params / 10**6, 2), "Million")
    
    data_dir = "/mnt/t9/video_latents"
    dataset = LatentDataset(data_dir, augment = True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn)
    maxlen = 0
    lengths = []
    for batch in dataloader:
        # Process the batch
        torch.set_printoptions(threshold=float('inf'), linewidth=99999)
        start = time.perf_counter()
        latent_tensor = batch["latent_tensor"].to(device) # Tensor to compress
        split_attn_mask = batch["split_attn_mask"].to(device) # attention mask after chunking video
        cutoffs = batch["cutoffs"].to(device) # (Batch, length) vector 
        cutoff_answer_mask = batch["cutoff_answer_mask"].to(device) # Mask to suppress loss from extra dims, in shape (batch, length)
        global_attention_mask = batch["global_attention_mask"].to(device) # Mask to strip padding
        compression_mask = batch["compression_mask"].to(device) # Mask out all but the first and last frames of each chunk
        decompression_mask = batch["decompression_mask"].to(device) # Mask that moves the first frame accross chunks
        if latent_tensor.shape[1] >= 300:
            continue
        reconstruction, chunks = model.forward(latent_tensor, split_attn_mask, global_attention_mask, compression_mask, decompression_mask)
        print(time.perf_counter() - start)
        exit()
        