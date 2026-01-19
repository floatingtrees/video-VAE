#!/usr/bin/env python3
"""
Script to check all videos the dataloader tries to open and report any that cannot be opened.
"""

import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataloader import list_video_files
from dataloader import load_video

NUM_THREADS = 32


def check_video(path: str) -> tuple[str, bool]:
    """
    Check if a video can be opened and has at least one readable frame.
    Returns (path, True) if video is valid, (path, False) otherwise.
    """
    try:
        cap = cv2.VideoCapture(path)
        v, m = load_video(path, 8, (256, 256), 512)
        return (path, True)
    except Exception:
        return (path, False)


def check(path):
    video_paths = list_video_files(path)
    print(f"Checking {len(video_paths)} videos with {NUM_THREADS} threads...")

    failed_videos = []
    completed = 0

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(check_video, p): p for p in video_paths}

        for future in as_completed(futures):
            completed += 1
            if completed % 1000 == 0:
                print(f"Progress: {completed}/{len(video_paths)}", flush=True)

            video_path, success = future.result()
            if not success:
                failed_videos.append(video_path)

    print(f"\n--- Failed videos ({len(failed_videos)}) ---")
    for failed_path in failed_videos:
        print(failed_path)

    print(f"\nDone. {len(failed_videos)} videos failed out of {len(video_paths)}")


if __name__ == "__main__":
    v, m = load_video("./", 3, (256, 256), 512)
    #print(v)
    #print(m)
    #print(v.shape, m.shape)
    check("/mnt/t9/videos_eval")
    check("/mnt/t9/videos")
    
