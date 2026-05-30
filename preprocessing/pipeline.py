import cv2
import numpy as np
import os
from loader import load_raw_data
from calibration import calibrate_frame
from glare import process_glare

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SIZE = (128, 128)
RAW_DIR     = r"D:\PupilTrackNN\data\raw"
CSV_PATH    = r"D:\PupilTrackNN\data\raw\camera_log.csv"
OUTPUT_DIR  = r"D:\PupilTrackNN\data\processed"

def to_grayscale(image):
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize(image):
    """Normalize pixel values using CLAHE for better local contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def resize(image):
    """Resize image to TARGET_SIZE."""
    return cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)

def process_frame(image, csv_row):
    """
    Full preprocessing pipeline for a single frame.
    Steps: calibrate → grayscale → normalize → resize → glare detection

    Returns:
        processed_image: clean 128x128 grayscale frame
        glare_mask:      128x128 binary mask (255 = glare, 0 = clean)
        glare_count:     number of glare spots found
    """
    img = calibrate_frame(image, csv_row)
    img = to_grayscale(img)
    img = normalize(img)
    img = resize(img)
    img, glare_mask, glare_count = process_glare(img)

    return img, glare_mask, glare_count

def run_pipeline():
    """
    Load all raw data, process each frame, save image + glare mask.
    
    Output per frame:
        data/processed/frame_XXXX.png        ← clean preprocessed image
        data/processed/frame_XXXX_glare.png  ← glare mask for NN
    """
    # Create output subfolders
    images_dir = os.path.join(OUTPUT_DIR, "images")
    masks_dir  = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)

    data = load_raw_data(RAW_DIR, CSV_PATH)

    if not data:
        print("[pipeline] No frames found. Add images to data/raw/ first.")
        return

    total_glare = 0

    for frame_id, image, csv_row in data:
        processed, glare_mask, glare_count = process_frame(image, csv_row)
        total_glare += glare_count

        # Save processed image
        img_path = os.path.join(images_dir, f"frame_{frame_id:04d}.png")
        cv2.imwrite(img_path, processed)

        # Save glare mask alongside it
        mask_path = os.path.join(masks_dir, f"frame_{frame_id:04d}_glare.png")
        cv2.imwrite(mask_path, glare_mask)

        print(f"[pipeline] Frame {frame_id:04d} saved → image + mask")

    print(f"\n[pipeline] Done.")
    print(f"           Frames processed : {len(data)}")
    print(f"           Total glare spots: {total_glare}")
    print(f"           Output           : {OUTPUT_DIR}")

if __name__ == "__main__":
    run_pipeline()