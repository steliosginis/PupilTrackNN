import cv2
import os

def load_images(raw_dir):
    """
    Load all images from raw_dir.
    Returns list of (frame_id, image) tuples.
    Supports .jpg, .png, .bmp
    """
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(supported)
    ])

    data = []
    for i, filename in enumerate(files):
        path = os.path.join(raw_dir, filename)
        img  = cv2.imread(path)
        if img is None:
            print(f"[loader] Could not read {filename}, skipping")
            continue
        data.append((i, img))
        print(f"[loader] Loaded {filename} → frame_id={i}")

    print(f"[loader] Total loaded: {len(data)} images")
    return data