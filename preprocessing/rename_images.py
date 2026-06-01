import os

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_DIR = r"D:\PupilTrackNN\data\raw\pupil_images\pupil_images"

def rename_images(input_dir):
    """
    Rename all image files in input_dir to frame_000.tiff, frame_001.tiff, etc.
    Sorted alphabetically before renaming to preserve original order.
    """
    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(supported)
    ])

    if not files:
        print(f"[rename] No images found in {input_dir}")
        return

    print(f"[rename] Found {len(files)} images — renaming...")

    for i, filename in enumerate(files):
        ext      = os.path.splitext(filename)[1].lower()
        new_name = f"frame_{i:03d}{ext}"
        old_path = os.path.join(input_dir, filename)
        new_path = os.path.join(input_dir, new_name)

        if old_path == new_path:
            continue

        os.rename(old_path, new_path)
        print(f"  {filename} → {new_name}")

    print(f"\n[rename] Done. {len(files)} files renamed.")

if __name__ == "__main__":
    rename_images(INPUT_DIR)