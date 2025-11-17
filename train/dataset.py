import torch
from torch.utils.data import DataLoader, Dataset
import os
import time
from random import random
def list_files(dir_path):
    return [
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]

def generate_attn_mask(input_list, seq_len):
    mask = torch.zeros(seq_len, seq_len)
    mask.fill_(float('-inf'))
    input_list = [0] + input_list + [seq_len, ]
    for i in range(1, len(input_list)):
        start = input_list[i-1]
        end = input_list[i]
        mask[start:end, start:end] = 0
    return mask
        
def generate_decompression_mask(input_list, seq_len):
    mask = torch.zeros((seq_len, seq_len))
    input_list = [0] + input_list + [seq_len, ]
    for i in range(1, len(input_list)):
        start = input_list[i-1]
        end = input_list[i]
        mask[start, start:end] = 1
        interpolation = torch.linspace(0.0, 1.0, steps=end - start) 
        mask[end-1, start:end] = interpolation
    return mask

class LatentDataset(Dataset):
    def __init__(self, data_dir, max_len = 300, augment = False):
        self.filenames = list_files(data_dir)
        self.data_dir = data_dir
        self.augment = augment
        self.max_len = max_len

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filepath = self.filenames[idx]
        data = torch.load(os.path.join(self.data_dir, filepath))
        latents = data["latents"][:self.max_len, ...]
        seq_len = latents.shape[0]
        hist_diff_list_old = data["hist_diff_list"]
        hist_diff_list = []
        for element in hist_diff_list_old:
            if element < self.max_len:
                hist_diff_list.append(element)
        mask_list = [0, seq_len - 1]
        if self.augment: # add some noise to the labels
            iteration_list = [0] + hist_diff_list + [seq_len]
            for i in range(1, len(iteration_list) - 1):
                #print(iteration_list)
                left = iteration_list[i-1]
                center = iteration_list[i]
                right = iteration_list[i + 1]
                right_deviation = (right - center) // 8
                left_deviation = (center - left) // 8
                randomval = torch.clamp(torch.randn(1), -2, 2).item()
                if randomval < 0:
                    randomval *= left_deviation
                else:
                    randomval *= right_deviation
                hist_diff_list[i-1] = center + int(randomval) # Iteration list prepends a 0
        for element in hist_diff_list:
            mask_list.append(element - 1)
            mask_list.append(element)
        split_attn_mask = generate_attn_mask(hist_diff_list, seq_len)
        decompression_mask = generate_decompression_mask(hist_diff_list, seq_len)
        return {"latents": latents, "split_attn_mask": split_attn_mask, "hist_diff_list": hist_diff_list, 
                "mask_list": mask_list, "decompression_mask": decompression_mask}
      
# NEED TO IMPLEMENT PACKING
def collate_fn(batch):
    batch_size = len(batch)
    channels, height, width = batch[0]["latents"].shape[-3], batch[0]["latents"].shape[-2], batch[0]["latents"].shape[-1]
    max_length = 0
    for item in batch:
        length = item["latents"].shape[0]
        max_length = max(max_length, length)
    latent_tensor = torch.zeros((batch_size, max_length, channels, height, width)) # latents from VAE
    split_attn_mask = torch.zeros((batch_size, max_length, max_length)) # attention mask for frames we split
    decompression_mask = torch.zeros((batch_size, max_length, max_length))
    cutoffs = torch.zeros((batch_size, max_length)) # labels for chunker predictions
    cutoff_answer_mask = torch.zeros((batch_size, max_length)) # multiply this with the cutoffs before computing loss
    
    # Mask for attention when detective cutoffs so padding doesn't matter
    global_attention_mask = torch.zeros((batch_size, max_length, max_length)) 
    global_attention_mask.fill_(float("-inf"))
    split_attn_mask.fill_(float('-inf'))
    compression_mask = torch.zeros((batch_size, max_length))
    total_parts = 0
    for i, item in enumerate(batch):
        sample_latent = item["latents"]
        sample_split_attn_mask = item["split_attn_mask"]
        sample_hist_diff_list = item["hist_diff_list"]
        compression_mask_list = item["mask_list"]
        sample_decompression_mask = item["decompression_mask"]
        
        length = sample_latent.shape[0]
        total_parts += length
        latent_tensor[i, :length, :, :, :] = sample_latent
        split_attn_mask[i, :length, :length] = sample_split_attn_mask
        decompression_mask[i, :length, :length] = sample_decompression_mask
        global_attention_mask[i, :length, :length] = 0
        
        split_attn_mask[i, length:, 0] = 0 # 0 out the tails so we don't get nans
        global_attention_mask[i, length:, 0] = 0
        cutoff_answer_mask[i, :length] = 1
        for element in sample_hist_diff_list:
            cutoffs[i, element] = 1
        for element in compression_mask_list:
            compression_mask[i, element] = 1
    return {"latent_tensor": latent_tensor, "split_attn_mask": split_attn_mask, "cutoffs": cutoffs,
            "cutoff_answer_mask": cutoff_answer_mask, "global_attention_mask": global_attention_mask, 
            "compression_mask": compression_mask, "decompression_mask": decompression_mask, 
            "total_parts": total_parts}
        
    
  
if __name__ == "__main__":
    data_dir = "/mnt/t9/video_latents"
    dataset = LatentDataset(data_dir, augment = True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    maxlen = 0
    lengths = []
    for batch in dataloader:
        # Process the batch
        torch.set_printoptions(threshold=float('inf'), linewidth=99999)
        latent_tensor = batch["latent_tensor"]
        split_attn_mask = batch["split_attn_mask"]
        cutoffs = batch["cutoffs"]
        cutoff_answer_mask = batch["cutoff_answer_mask"]
        global_attention_mask = batch["global_attention_mask"]
        compression_mask = batch["compression_mask"]
        decompression_mask = batch["decompression_mask"]
        #print(cutoffs)
        #print(compression_mask)
        for i in range(compression_mask.shape[1]):
            if compression_mask[0, i].item() == 1:
                split_attn_mask[0, i, 0] = 1
        #print(split_attn_mask)
        print(decompression_mask)
        exit()
    print("Max length: ", maxlen)
    from statistics import mean, stdev
    print("Mean length: ", mean(lengths))
    print("Std Dev length: ", stdev(lengths))