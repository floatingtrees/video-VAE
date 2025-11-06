import cv2
import numpy as np
import torch
from PIL import Image
import gc
import json, shutil, subprocess
import cv2
from PIL import Image
from typing import List

import cv2
import numpy as np
from PIL import Image
from typing import List
from efficientvit.ae_model_zoo import DCAE_HF
from torchvision import transforms
from efficientvit.apps.utils.image import DMCrop
import time
import sys

def get_fps(path: str):
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        cap.release()
        if fps > 0:
            return fps

    # Fallback to ffprobe if available
    if shutil.which("ffprobe"):
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path
            ])
            data = json.loads(out)
            for s in data.get("streams", []):
                if s.get("codec_type") == "video":
                    for key in ("avg_frame_rate", "r_frame_rate"):
                        val = s.get(key)
                        if val and val != "0/0":
                            num, den = map(int, val.split("/"))
                            if den != 0:
                                return num / den
        except Exception:
            pass
    return None


def convertToImage(tensor):
    image = tensor[0, :, :, :].detach().numpy()
    image = np.clip((np.transpose(image, (1, 2, 0)) + 1) / 2, 0, 1)
    return Image.fromarray((image * 255).round().astype(np.uint8))



def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def hist_diff_indices_pil(images: List[Image.Image],
                          hist_size: int = 64,
                          similarity_thresh: float = 0.85,
                          use_hsv: bool = True):
    """
    images: list of PIL.Image objects
    hist_size: number of bins
    similarity_thresh: correlation below this => significant change
    use_hsv: if True, compute hist on HSV; else on grayscale

    returns: list of indices i where frame i is a change point vs i-1
    """
    if not images:
        return []

    change_indices = []
    prev_hist = None
    error_accum = 0.0
    for i, pil_img in enumerate(images):
        # PIL -> RGB numpy
        rgb = np.array(pil_img.convert("RGB"))

        if use_hsv:
            # RGB -> HSV (OpenCV wants BGR, so convert properly)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist(
                [hsv], [0, 1], None,
                [hist_size, hist_size],
                [0, 180, 0, 256]
            )
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist(
                [gray], [0], None,
                [hist_size],
                [0, 256]
            )

        hist = cv2.normalize(hist, None).flatten()

        if prev_hist is not None:
            sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            error_accum += (1 - sim)
            if error_accum > (1 - similarity_thresh):
                change_indices.append(i)
                error_accum = 0

        prev_hist = hist

    return change_indices


def video_to_pil_list(path: str, *, stride: int = 1, limit: int | None = None,
                      convert_mode: str | None = None) -> List[Image.Image]:
    """
    Read a video and return a list of PIL.Image objects (RGB by default).

    Args:
        path: Path to the video file.
        stride: Keep 1 frame out of every `stride` frames (>=1).
        limit: If set, stop after collecting this many frames.
        convert_mode: Optional PIL mode to convert each frame to (e.g., "L", "RGBA").

    Returns:
        List of PIL.Image objects (possibly empty).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")

    images: List[Image.Image] = []
    i = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if (i % stride) == 0:
                # BGR -> RGB for PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)  # mode "RGB"
                if convert_mode:
                    img = img.convert(convert_mode)
                images.append(img)

                if limit is not None and len(images) >= limit:
                    break
            i += 1
    finally:
        cap.release()

    return images

import os

def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# Example
STORAGE_PATH = "/mnt/t9/videos/videos2"
files = list_files(STORAGE_PATH)
print(files[0])



dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f64c128-in-1.0").to(torch.bfloat16)
device = torch.device("cuda")
dc_ae = dc_ae.to(device).eval()
transform = transforms.Compose([
    DMCrop(512), # resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

BATCH_SIZE = 64


save_dir = "/mnt/t9/video_latents"
start = time.perf_counter()
with torch.no_grad():
    for counter, filepath in enumerate(files):
        try:
            tensor_cat_list = []
            #fps = get_fps(f"{STORAGE_PATH}/{filepath}")

            arr = video_to_pil_list(f"{STORAGE_PATH}/{filepath}")
            
            hist_diff_list = hist_diff_indices_pil(arr)
            for element in arr:
                tensor_cat_list.append(transform(element).unsqueeze(0))
            out_list = []
            for i in range(0, len(tensor_cat_list), BATCH_SIZE):
                batch = torch.cat(tensor_cat_list[i:min(i+BATCH_SIZE, len(tensor_cat_list))], dim=0)
                encoded = dc_ae.encode(batch.to(device).to(torch.bfloat16))
                out_list.append(encoded)
            latents = torch.cat(out_list, dim=0).to("cpu")
            torch.save({"latents" : latents, "hist_diff_list": hist_diff_list}, f"{save_dir}/{filepath}.pt")
            #out_img = convertToImage(decoded.float())
            #out_img.save("./test2.png")
            if counter % 100 == 0:
                print(counter, time.perf_counter() - start)
            sys.stdout.flush()
        except Exception as e:
            print(e)
print("DONE PROCESSING")