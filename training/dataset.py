import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ANNOTATED_DIR = r"D:\PupilTrackNN\data\annotated"
LABELS_CSV    = r"D:\PupilTrackNN\data\annotated\labels.csv"

# ── Noise functions ───────────────────────────────────────────────────────────

def no_noise(image):
    return image

def gaussian_noise(image, std=10):
    noise  = np.random.normal(0, std, image.shape).astype(np.float32)
    result = np.clip(image.astype(np.float32) + noise, 0, 255)
    return result.astype(np.uint8)

def salt_and_pepper(image, amount=0.02):
    result  = image.copy()
    n_salt  = int(amount * image.size * 0.5)
    n_pepper= int(amount * image.size * 0.5)
    # Salt
    coords  = [np.random.randint(0, i, n_salt) for i in image.shape]
    result[coords[0], coords[1]] = 255
    # Pepper
    coords  = [np.random.randint(0, i, n_pepper) for i in image.shape]
    result[coords[0], coords[1]] = 0
    return result

def brightness_shift(image, delta=20):
    shifted = np.clip(image.astype(np.float32) + np.random.uniform(-delta, delta), 0, 255)
    return shifted.astype(np.uint8)

def blur_variation(image):
    k = np.random.choice([3, 5])
    return cv2.GaussianBlur(image, (k, k), 0)

def center_shift(image, max_shift=10):
    """
    Shift the image 1-2 pixels in a random direction (up/down/left/right).
    Simulates small camera jitter between frames.
    """
    dx = np.random.randint(-max_shift, max_shift + 3)
    dy = np.random.randint(-max_shift, max_shift + 3)

    M      = np.float32([[1, 0, dx], [0, 1, dy]])
    result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return result

def combine_noise(image, modes):
    """Apply multiple noise functions in sequence."""
    result = image.copy()
    for mode in modes:
        result = NOISE_MODES[mode](result)
    return result

NOISE_MODES = {
    "clean":                    no_noise,
    "gaussian":                 gaussian_noise,
    "salt_and_pepper":          salt_and_pepper,
    "brightness":               brightness_shift,
    "blur":                     blur_variation,
    "center_shift":             center_shift,
    "gaussian+center_shift":    lambda img: combine_noise(img, ["gaussian", "center_shift"]),
    "blur+brightness":          lambda img: combine_noise(img, ["blur", "brightness"]),
    "gaussian+salt_and_pepper": lambda img: combine_noise(img, ["gaussian", "salt_and_pepper"]),
    "all":                      lambda img: combine_noise(img, ["gaussian", "salt_and_pepper",
                                                                 "brightness", "blur", "center_shift"]),
}

# ── Dataset ───────────────────────────────────────────────────────────────────

class PupilDataset(Dataset):
    def __init__(self, frame_ids, labels_df, noise_mode="clean"):
        """
        Args:
            frame_ids:  list of frame_id values for this split
            labels_df:  full labels dataframe
            noise_mode: one of NOISE_MODES keys
        """
        self.labels    = labels_df[labels_df["frame_id"].isin(frame_ids)].reset_index(drop=True)
        self.noise_fn  = NOISE_MODES[noise_mode]
        self.noise_mode= noise_mode

        if len(self.labels) == 0:
            raise ValueError(f"[dataset] No labels found for given frame_ids")

        print(f"[dataset] {len(self.labels)} samples | noise={noise_mode}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row      = self.labels.iloc[idx]
        frame_id = int(row["frame_id"])

        img_path = os.path.join(ANNOTATED_DIR, f"frame_{frame_id:04d}.png")
        img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"[dataset] Image not found: {img_path}")

        img = self.noise_fn(img)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

    # Hough hint — normalized to 0-1
        hough_hint = torch.tensor(
            [row["cx"] / 128.0, row["cy"] / 128.0],
            dtype=torch.float32
        )

    # Label is the same as hint for clean data
    # For noisy data the NN learns to correct from the hint
        label = torch.tensor(
            [row["cx"] / 128.0, row["cy"] / 128.0],
            dtype=torch.float32
        )

        return img_tensor, hough_hint, label


def get_splits(labels_csv=LABELS_CSV, train=0.80, val=0.15):
    """
    Split frame_ids into train / val / test sets.
    Returns three lists of frame_ids.
    """
    df         = pd.read_csv(labels_csv)
    frame_ids  = df["frame_id"].tolist()
    n          = len(frame_ids)

    n_train    = int(n * train)
    n_val      = int(n * val)

    train_ids  = frame_ids[:n_train]
    val_ids    = frame_ids[n_train:n_train + n_val]
    test_ids   = frame_ids[n_train + n_val:]

    print(f"[dataset] Split → train={len(train_ids)} "
          f"val={len(val_ids)} test={len(test_ids)}")

    return train_ids, val_ids, test_ids