import cv2
import numpy as np
import os
import csv
from loader import load_raw_data
from calibration import calibrate_frame
from glare import process_glare
from roi import find_pupil_center , draw_detection

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SIZE    = (128, 128)
RAW_DIR        = r"D:\PupilTrackNN\data\raw"
CSV_PATH       = r"D:\PupilTrackNN\data\raw\camera_log.csv"
PROCESSED_DIR  = r"D:\PupilTrackNN\data\processed"
ANNOTATED_DIR  = r"D:\PupilTrackNN\data\annotated"
LABELS_CSV     = r"D:\PupilTrackNN\data\annotated\labels.csv"

def crop_to_square(image):
    """
    Center crop the image to a square before resizing.
    Takes the largest possible square from the center of the frame.
    Preserves aspect ratio — no squishing or stretching.
    """
    h, w = image.shape[:2]
    side = min(h, w)

    # Calculate crop boundaries centered on the image
    top  = (h - side) // 2
    left = (w - side) // 2

    return image[top:top+side, left:left+side]

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

def preprocess(image, csv_row):
    """
    Core preprocessing steps shared by both modes.
    Steps: calibrate → grayscale → normalize → resize
    """
    img = calibrate_frame(image, csv_row)
    img = crop_to_square(img) 
    img = to_grayscale(img)
    img = normalize(img)
    img = resize(img)
    return img

def run_labeling_pipeline():
    """
    Mode 1 — Auto-labeling pipeline for clean images.
    Uses Hough to find pupil center and saves:
        data/annotated/<frame_id>.png   ← preprocessed image
        data/annotated/labels.csv       ← frame_id, cx, cy
    Frames where Hough fails are discarded.
    """
    os.makedirs(ANNOTATED_DIR, exist_ok=True)

    data = load_raw_data(RAW_DIR, CSV_PATH)

    if not data:
        print("[pipeline] No frames found. Add images to data/raw/ first.")
        return

    saved    = 0
    discarded = 0

    with open(LABELS_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_id", "cx", "cy", "radius"])

        for frame_id, image, csv_row in data:
            img = preprocess(image, csv_row)

            center, radius = find_pupil_center(img)

            if center is None:
                print(f"[pipeline] Frame {frame_id:04d} → DISCARDED (no circle found)")
                discarded += 1
                continue

            cx, cy = center

            # Save annotated image
            # Save preprocessed image
            img_path = os.path.join(ANNOTATED_DIR, f"frame_{frame_id:04d}.png")
            cv2.imwrite(img_path, img)

            # Save verification image with circle and center drawn
            vis = draw_detection(img, center, radius)
            vis_path = os.path.join(ANNOTATED_DIR, f"frame_{frame_id:04d}_detection.png")
            cv2.imwrite(vis_path, vis)

            # Write label row
            writer.writerow([frame_id, cx, cy, radius])
            saved += 1

            print(f"[pipeline] Frame {frame_id:04d} → center=({cx}, {cy}) "
                  f"radius={radius}px ✓ saved")

    print(f"\n[pipeline] Labeling done.")
    print(f"           Saved    : {saved} frames")
    print(f"           Discarded: {discarded} frames")
    print(f"           Labels   : {LABELS_CSV}")

def run_inference_pipeline():
    """
    Mode 2 — Inference pipeline for real/noisy images.
    Runs full preprocessing including glare detection.
    NN prediction happens separately.
    """
    images_dir = os.path.join(PROCESSED_DIR, "images")
    masks_dir  = os.path.join(PROCESSED_DIR, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)

    data = load_raw_data(RAW_DIR, CSV_PATH)

    if not data:
        print("[pipeline] No frames found.")
        return

    for frame_id, image, csv_row in data:
        img = preprocess(image, csv_row)
        img, glare_mask, glare_count = process_glare(img)

        img_path  = os.path.join(images_dir, f"frame_{frame_id:04d}.png")
        mask_path = os.path.join(masks_dir,  f"frame_{frame_id:04d}_glare.png")
        cv2.imwrite(img_path,  img)
        cv2.imwrite(mask_path, glare_mask)

        print(f"[pipeline] Frame {frame_id:04d} → glare spots={glare_count}")

    print(f"\n[pipeline] Inference preprocessing done → {PROCESSED_DIR}")

if __name__ == "__main__":
    print("Select mode:")
    print("  1 - Labeling pipeline (clean images → auto-label with Hough)")
    print("  2 - Inference pipeline (real images → preprocess for NN)")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        run_labeling_pipeline()
    elif mode == "2":
        run_inference_pipeline()
    else:
        print("Invalid choice. Enter 1 or 2.")