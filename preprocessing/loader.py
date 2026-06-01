import cv2
import os

RAW_DIR = r"D:\PupilTrackNN\data\raw\pupil_images\pupil_images"

def load_images(raw_dir=RAW_DIR):
    """
    Load all images from raw_dir, downsampled to save memory.
    Returns list of (frame_id, image) tuples.
    """
    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(supported)
    ])

    data = []
    for i, filename in enumerate(files):
        path = os.path.join(raw_dir, filename)

        # Load at 1/4 resolution to save memory — still plenty for 128x128 output
        img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)

        if img is None:
            print(f"[loader] Could not read {filename}, skipping")
            continue

        data.append((i, img))
        print(f"[loader] {filename} → {img.shape[1]}x{img.shape[0]}")

    print(f"[loader] Loaded {len(data)} images")
    return data