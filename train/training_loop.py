import torch 
from model import VideoEncoder
from dataset import LatentDataset, collate_fn
from torch.utils.data import DataLoader
import time
import wandb
from torch.optim.lr_scheduler import LambdaLR
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
import torch.nn.functional as F
from einops import rearrange
wandb.init(project="video encoder")

GRAD_ACCUM_STEPS = 32
BATCH_SIZE = 16
MAX_LEN = 200
WARMUP_STEPS = 1000

def linear_schedule(step):
    step += 1
    
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    return 1.0

device = "cuda"
model = VideoEncoder(channels=128, height=8, width=8, depth=4, temporal_dim = 1024)
model.to(device)
model.train()
total_params = 0
for element in model.parameters():
    total_params += element.numel()
print("Total number of parameters: ", round(total_params / 10**6, 2), "Million")

data_dir = "/mnt/t9/video_latents"
dataset = LatentDataset(data_dir, max_len = 200, augment = True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, collate_fn=collate_fn)
maxlen = 0
lengths = []
start = time.perf_counter()
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4)
scheduler = LambdaLR(optimizer, lr_lambda=linear_schedule)
for j in range(100):
    for i, batch in enumerate(dataloader):
        start2 = time.perf_counter()
        latent_tensor = batch["latent_tensor"].to(device) # Tensor to compress
        split_attn_mask = batch["split_attn_mask"].to(device) # attention mask after chunking video
        cutoffs = batch["cutoffs"].to(device) # (Batch, length) vector 
        cutoff_answer_mask = batch["cutoff_answer_mask"].to(device) # Mask to suppress loss from extra dims, in shape (batch, length)
        global_attention_mask = batch["global_attention_mask"].to(device) # Mask to strip padding
        compression_mask = batch["compression_mask"].to(device) # Mask out all but the first and last frames of each chunk
        decompression_mask = batch["decompression_mask"].to(device) # Mask that moves the first frame accross chunks
        total_parts = batch["total_parts"] # Number of nonmasked sequence parts
        b, s, c, h, w  = latent_tensor.shape
        unmasked_elements = b * c * h * w * total_parts
        
        reconstruction, chunk_logits = model.forward(latent_tensor, split_attn_mask, global_attention_mask, compression_mask, decompression_mask)
        chunk_logits = chunk_logits.squeeze(-1)
        chunking_diff = F.binary_cross_entropy_with_logits(chunk_logits, cutoffs, reduction="none")
        chunking_loss = torch.sum(chunking_diff * cutoff_answer_mask) / (b * total_parts)
        
        reconstruction_diff = torch.square(latent_tensor - reconstruction)

        cutoff_mask_view = cutoff_answer_mask.view(cutoff_answer_mask.shape[0], cutoff_answer_mask.shape[1], 1, 1, 1)
        reconstruction_loss = torch.sum(reconstruction_diff * cutoff_mask_view) / unmasked_elements
        
        loss = (reconstruction_loss + chunking_loss) / GRAD_ACCUM_STEPS
        loss.backward()

        wandb.log({"time": time.perf_counter() - start, "reconstruction_loss": reconstruction_loss.item(), 
                    "chunking_loss": chunking_loss.item(), "loss" : loss.item(), 
                    "lr": optimizer.param_groups[0]["lr"]})

        if i % GRAD_ACCUM_STEPS == (GRAD_ACCUM_STEPS - 1):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()