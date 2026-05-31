import cv2
import numpy as np
import os
import csv
from loader import load_images

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SIZE   = (128, 128)
RAW_DIR       = r"D:\PupilTrackNN\data\raw"
ANNOTATED_DIR = r"D:\PupilTrackNN\data\annotated"
LABELS_CSV    = r"D:\PupilTrackNN\data\annotated\labels.csv"

# ── Steps ────────────────────────────────────────────────────────────────────

def crop_to_square(image):
    """Center crop to square — no squishing or stretching."""
    h, w = image.shape[:2]
    side  = min(h, w)
    top   = (h - side) // 2
    left  = (w - side) // 2
    return image[top:top+side, left:left+side]

def to_grayscale(image):
    """Convert BGR to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize(image):
    """Resize to 128x128."""
    return cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)

def apply_gaussian(image):
    """Blur to reduce noise before CLAHE and Hough."""
    return cv2.GaussianBlur(image, (7, 7), 0)

def apply_clahe(image):
    """Enhance local contrast to make the pupil circle more prominent."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def find_pupil(image, min_radius=5, max_radius=15):
    """
    Use Hough Circle Transform to find the pupil.
    Expects a preprocessed 128x128 grayscale image.

    Returns:
        center: (x, y) or None
        radius: int or None
    """
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=10,
        param2=5,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return None, None

    circles = np.round(circles[0, :]).astype(int)

    # Pick the darkest circle — pupil is always darkest under IR
    best_circle  = None
    darkest_mean = 255

    for (x, y, r) in circles:
        mask = np.zeros_like(image)
        cv2.circle(mask, (x, y), r, 255, -1)
        mean_brightness = cv2.mean(image, mask=mask)[0]
        if mean_brightness < darkest_mean:
            darkest_mean = mean_brightness
            best_circle  = (x, y, r)

    cx, cy, radius = best_circle
    return (cx, cy), radius

def draw_detection(image, center, radius):
    """
    Draw detected circle (green) and center point (red) on the image.
    Returns a BGR image for saving.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cx, cy = center
    cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 1)  # green circle
    cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)       # red center dot
    return vis

# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_frame(image):
    """
    Full labeling pipeline for one frame.

    Steps: crop → grayscale → resize → gaussian → clahe → hough

    Returns:
        processed:  128x128 grayscale image (clean, pre-Hough)
        vis:        128x128 BGR image with circle + center drawn
        center:     (x, y) or None
        radius:     int or None
    """
    img = crop_to_square(image)
    img = to_grayscale(img)
    img = resize(img)
    img = apply_gaussian(img)
    img = apply_clahe(img)

    center, radius = find_pupil(img)

    if center is None:
        return None, None, None, None

    vis = draw_detection(img, center, radius)
    return img, vis, center, radius
def run_pipeline():
    """
    Load all raw frames, process each one.
    Saves to data/annotated/:
        frame_XXXX.png           ← clean processed image
        frame_XXXX_detection.png ← image with circle + center drawn
        labels.csv               ← frame_id, cx, cy
    Frames where Hough fails are silently discarded.
    """
    os.makedirs(ANNOTATED_DIR, exist_ok=True)

    data = load_images(RAW_DIR)

    if not data:
        print("[pipeline] No images found in data/raw/")
        return

    saved     = 0
    discarded = 0

    with open(LABELS_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_id", "cx", "cy"])

        for frame_id, image in data:
            img, vis, center, radius = process_frame(image)

            if center is None:
                print(f"[pipeline] Frame {frame_id:04d} → no pupil found, discarded")
                discarded += 1
                continue

            cx, cy = center

            # Save clean processed image
            img_path = os.path.join(ANNOTATED_DIR, f"frame_{frame_id:04d}.png")
            cv2.imwrite(img_path, img)

            # Save detection visualization
            vis_path = os.path.join(ANNOTATED_DIR, f"frame_{frame_id:04d}_detection.png")
            cv2.imwrite(vis_path, vis)

            # Write label
            writer.writerow([frame_id, cx, cy])
            saved += 1

            print(f"[pipeline] Frame {frame_id:04d} → center=({cx},{cy}) radius={radius}px ✓")

    print(f"\n[pipeline] Done.")
    print(f"           Saved    : {saved}")
    print(f"           Discarded: {discarded}")
    print(f"           Labels   : {LABELS_CSV}")

if __name__ == "__main__":
    run_pipeline()