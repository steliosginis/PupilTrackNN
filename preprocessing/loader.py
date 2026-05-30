import cv2
import pandas as pd
import os

def load_csv(csv_path):
    """Load camera log CSV and return as DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"[loader] Loaded {len(df)} rows from {csv_path}")
    return df

def load_image(image_path):
    """Load a single image and return as numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[loader] Could not load image: {image_path}")
    return img

def load_video_frames(video_path):
    """Extract all frames from a video file, return as list."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[loader] Extracted {len(frames)} frames from {video_path}")
    return frames

def load_raw_data(raw_dir, csv_path):
    """
    Load all images/videos from raw_dir and match with CSV rows.
    Returns list of (frame_id, image, csv_row) tuples.
    """
    df = load_csv(csv_path)
    data = []

    for _, row in df.iterrows():
        frame_id = int(row['frame_id'])
        # Look for matching image file
        for ext in ['.jpg', '.png', '.bmp']:
            img_path = os.path.join(raw_dir, f"frame_{frame_id:04d}{ext}")
            if os.path.exists(img_path):
                img = load_image(img_path)
                data.append((frame_id, img, row))
                break

    print(f"[loader] Matched {len(data)} frames with CSV rows")
    return data