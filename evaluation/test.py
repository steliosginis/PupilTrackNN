import torch
import cv2
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(r"D:\PupilTrackNN\training")
from model   import PupilNet
from dataset import get_splits

# ── Config ───────────────────────────────────────────────────────────────────
ANNOTATED_DIR = r"D:\PupilTrackNN\data\annotated"
LABELS_CSV    = r"D:\PupilTrackNN\data\annotated\labels.csv"
WEIGHTS_DIR   = r"D:\PupilTrackNN\output\weights"
RESULTS_DIR   = r"D:\PupilTrackNN\output\test_results"

def load_model(weights_path):
    model = PupilNet()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def predict(model, image, cx, cy):
    """
    Run inference with image + Hough hint.
    """
    img_tensor  = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    hough_hint  = torch.tensor([[cx / 128.0, cy / 128.0]], dtype=torch.float32)

    with torch.no_grad():
        output = model(img_tensor, hough_hint)

    pred_cx = output[0][0].item() * 128
    pred_cy = output[0][1].item() * 128
    return int(pred_cx), int(pred_cy)

def draw_result(image, true_center, pred_center):
    """
    Draw ground truth (green) and prediction (red) on the image.
    Green circle + dot = Hough ground truth
    Red circle + dot   = NN prediction
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tx, ty = true_center
    px, py = pred_center

    # Ground truth — green
    cv2.circle(vis, (tx, ty), 8, (0, 255, 0), 1)
    cv2.circle(vis, (tx, ty), 2, (0, 255, 0), -1)

    # Prediction — red
    cv2.circle(vis, (px, py), 8, (0, 0, 255), 1)
    cv2.circle(vis, (px, py), 2, (0, 0, 255), -1)

    # Line between them to show error distance
    cv2.line(vis, (tx, ty), (px, py), (255, 255, 0), 1)

    return vis

def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load test split
    _, _, test_ids = get_splits(LABELS_CSV)
    df = pd.read_csv(LABELS_CSV)
    test_df = df[df["frame_id"].isin(test_ids)].reset_index(drop=True)

    if len(test_df) == 0:
        print("[test] No test samples found.")
        return

    # Pick best available weights
    weight_files = sorted([
        f for f in os.listdir(WEIGHTS_DIR)
        if f.endswith(".pth")
    ])

    if not weight_files:
        print(f"[test] No weights found in {WEIGHTS_DIR}")
        return

    print("Available weight files:")
    for i, f in enumerate(weight_files):
        print(f"  {i+1}. {f}")
    choice = input("Enter weight file name to use: ").strip()

    weights_path = os.path.join(WEIGHTS_DIR, choice)
    if not os.path.exists(weights_path):
        print(f"[test] File not found: {weights_path}")
        return

    model = load_model(weights_path)
    print(f"\n[test] Loaded weights: {choice}")
    print(f"[test] Testing on {len(test_df)} images\n")

    errors = []

    for _, row in test_df.iterrows():
        frame_id = int(row["frame_id"])
        true_cx  = int(row["cx"])
        true_cy  = int(row["cy"])

        img_path = os.path.join(ANNOTATED_DIR, f"frame_{frame_id:04d}.png")
        img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[test] Could not load frame_{frame_id:04d}.png, skipping")
            continue

        pred_cx, pred_cy = predict(model, img, true_cx, true_cy)

        # Euclidean pixel error
        error = np.sqrt((pred_cx - true_cx)**2 + (pred_cy - true_cy)**2)
        errors.append(error)

        # Save result image
        vis      = draw_result(img, (true_cx, true_cy), (pred_cx, pred_cy))
        vis_big  = cv2.resize(vis, (512, 512), interpolation=cv2.INTER_NEAREST)
        out_path = os.path.join(RESULTS_DIR, f"frame_{frame_id:04d}_result.png")
        cv2.imwrite(out_path, vis_big)

        print(f"frame_{frame_id:04d} | "
              f"true=({true_cx},{true_cy}) "
              f"pred=({pred_cx},{pred_cy}) "
              f"error={error:.2f}px")

    # Summary
    errors = np.array(errors)
    print(f"\n{'─'*45}")
    print(f"Test results ({len(errors)} images)")
    print(f"  Mean error  : {errors.mean():.2f}px")
    print(f"  Median error: {np.median(errors):.2f}px")
    print(f"  Max error   : {errors.max():.2f}px")
    print(f"  Min error   : {errors.min():.2f}px")
    print(f"  < 2px       : {(errors < 2).sum()} / {len(errors)} frames")
    print(f"  < 5px       : {(errors < 5).sum()} / {len(errors)} frames")
    print(f"  < 10px      : {(errors < 10).sum()} / {len(errors)} frames")
    print(f"{'─'*45}")
    print(f"\n[test] Result images saved → {RESULTS_DIR}")

if __name__ == "__main__":
    run()