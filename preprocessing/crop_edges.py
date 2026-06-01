import cv2
import os

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_DIR = r"D:\PupilTrackNN\data\raw\pupil_images\pupil_images"

def crop_edges(image):
    """
    Crop the bottom 1/5 and right 1/5 of the image.
    Keeps the top-left 4/5 x 4/5 region where the eye is.
    """
    h, w  = image.shape[:2]
    new_h = int(h * 4 / 5)
    new_w = int(w * 4 / 5)
    return image[:new_h, :new_w]

def run(input_dir=INPUT_DIR):
    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(supported)
    ])

    if not files:
        print(f"[crop_edges] No images found in {input_dir}")
        return

    print(f"[crop_edges] Processing {len(files)} images...")

    for filename in files:
        path  = os.path.join(input_dir, filename)
        img   = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)

        if img is None:
            print(f"[crop_edges] Could not read {filename}, skipping")
            continue

        cropped = crop_edges(img)
        cv2.imwrite(path, cropped)
        print(f"[crop_edges] {filename} → {img.shape[1]}x{img.shape[0]} → {cropped.shape[1]}x{cropped.shape[0]}")

    print(f"\n[crop_edges] Done. {len(files)} images cropped in place.")

if __name__ == "__main__":
    run()