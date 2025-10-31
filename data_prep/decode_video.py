import torch
import subprocess, shutil, numpy as np
from PIL import Image
from typing import List
import numpy as np
from efficientvit.ae_model_zoo import DCAE_HF
from torchvision import transforms
from efficientvit.apps.utils.image import DMCrop

def convertToImage(tensor):
    image = tensor[0, :, :, :].float().detach().numpy()
    image = np.clip((np.transpose(image, (1, 2, 0)) + 1) / 2, 0, 1)
    return Image.fromarray((image * 255).round().astype(np.uint8))

def write_video_ffmpeg(frames: List[Image.Image], fps: float, out_path: str,
                       crf: int = 18, preset: str = "medium"):
    if not frames:
        raise ValueError("No frames to write.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    w, h = frames[0].size

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf), "-preset", preset,
        out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for img in frames:
            if img.size != (w, h):
                img = img.resize((w, h), Image.BICUBIC)
            rgb = np.asarray(img.convert("RGB"))
            proc.stdin.write(rgb.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg failed.")

BATCH_SIZE = 64
filepath = "/mnt/t9/video_latents/celebv_--Jiv5iYqT8_0.mp4.pt"
latents = torch.load(filepath)["latents"]
dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f64c128-in-1.0").to(torch.bfloat16)
device = torch.device("cuda")
dc_ae = dc_ae.to(device).eval()
video_len = latents.shape[0]
out_list = []
with torch.no_grad():
    for i in range(0, video_len, BATCH_SIZE):
        batch = latents[i:min(i+BATCH_SIZE, video_len)]
        encoded = dc_ae.decode(batch.to(device).to(torch.bfloat16))
        out_list.append(encoded)
reconstructed = torch.cat(out_list, dim=0).to("cpu")
frame_list = []
for i in range(video_len):
    frame = convertToImage(reconstructed[i:i+1])
    frame_list.append(frame)
write_video_ffmpeg(frame_list, fps=24, out_path="./test2.mp4")