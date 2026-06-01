import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(r"D:\PupilTrackNN\training")
from model   import PupilNet
from dataset import PupilDataset, get_splits, NOISE_MODES , combine_noise

# ── Config ───────────────────────────────────────────────────────────────────
WEIGHTS_DIR  = r"D:\PupilTrackNN\output\weights"
LABELS_CSV   = r"D:\PupilTrackNN\data\annotated\labels.csv"
EPOCHS       = 50
BATCH_SIZE   = 16
LR           = 1e-4

def train_round(noise_mode="clean"):
    """
    Train one round with the given noise mode.
    Saves best weights to output/weights/best_<noise_mode>.pth
    """
    if noise_mode not in NOISE_MODES:
        print(f"[train] Unknown noise mode. Choose from: {list(NOISE_MODES.keys())}")
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ids, val_ids, _ = get_splits(LABELS_CSV)

    train_set    = PupilDataset(train_ids, __import__('pandas').read_csv(LABELS_CSV), noise_mode)
    val_set      = PupilDataset(val_ids,   __import__('pandas').read_csv(LABELS_CSV), "clean")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model     = PupilNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Load previous best weights if they exist to continue training
    best_path = os.path.join(WEIGHTS_DIR, "best.pth") ## na to allajw 
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print(f"[train] Loaded existing weights from {best_path}")

    best_val_loss = float("inf")

    print(f"\n[train] Starting round: noise={noise_mode} | "
          f"epochs={EPOCHS} | batch={BATCH_SIZE}\n")

    for epoch in range(1, EPOCHS + 1):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for images, hints, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images, hints)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, hints, labels in val_loader:
                outputs   = model(images, hints)
                loss      = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Convert loss to pixel error for readability (loss is in 0-1 space)
        train_px = (train_loss ** 0.5) * 128
        val_px   = (val_loss   ** 0.5) * 128

        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"train={train_loss:.6f} ({train_px:.2f}px) | "
              f"val={val_loss:.6f} ({val_px:.2f}px)", end="")

        # Save best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  ← saved")
        else:
            print()

    print(f"\n[train] Round complete.")
    print(f"        Best val loss : {best_val_loss:.6f} "
          f"({(best_val_loss**0.5)*128:.2f}px)")
    print(f"        Weights saved : {best_path}")

if __name__ == "__main__":
    print("Available noise modes:")
    for i, mode in enumerate(NOISE_MODES.keys()):
        print(f"  {i+1}. {mode}")
    print("\nYou can also combine base modes with + (e.g. gaussian+blur+center_shift)")

    choice = input("\nEnter noise mode: ").strip()

    # If it's a custom combination not in NOISE_MODES, build it dynamically
    if choice not in NOISE_MODES:
        parts = choice.split("+")
        base_modes = ["clean", "gaussian", "salt_and_pepper",
                      "brightness", "blur", "center_shift"]
        invalid = [p for p in parts if p not in base_modes]
        if invalid:
            print(f"[train] Unknown mode(s): {invalid}")
            print(f"        Base modes: {base_modes}")
        else:
            # Register the combo dynamically
            NOISE_MODES[choice] = lambda img, p=parts: combine_noise(img, p)
            train_round(noise_mode=choice)
    else:
        train_round(noise_mode=choice)