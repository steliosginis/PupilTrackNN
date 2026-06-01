import pandas as pd
import numpy as np
import os

# ── Config ───────────────────────────────────────────────────────────────────
LABELS_CSV     = r"D:\PupilTrackNN\data\annotated\labels.csv"
OUTPUT_CSV     = r"D:\PupilTrackNN\data\annotated\labels_normalized.csv"
CHUNK_SIZE     = 150   # rows per chunk for mean X calculation
PHI_MULTIPLIER = 2.0   # φ = PHI_MULTIPLIER * std of X differences
                       # raise to be more lenient, lower to be stricter

def calculate_phi(df):
    """
    Automatically calculate φ from the data.
    φ = mean of absolute X differences + 2 standard deviations.
    Frames with X jump larger than φ are considered outliers.
    """
    x_diffs = df["cx"].diff().abs().dropna()
    phi     = x_diffs.mean() + PHI_MULTIPLIER * x_diffs.std()
    print(f"[normalize] X differences → mean={x_diffs.mean():.2f} "
          f"std={x_diffs.std():.2f}")
    print(f"[normalize] φ calculated  → {phi:.2f}px")
    return phi

def filter_outliers(df, phi):
    """
    Discard rows where the X jump from the previous valid frame
    exceeds φ. Discards both cx and cy together.
    """
    keep        = []
    last_valid_x = None

    for _, row in df.iterrows():
        cx = row["cx"]

        if last_valid_x is None:
            keep.append(True)
            last_valid_x = cx
        elif abs(cx - last_valid_x) > phi:
            keep.append(False)
            print(f"[normalize] Discarded frame_id={row['frame_id']} → "
                  f"X jump={abs(cx - last_valid_x):.1f}px > φ={phi:.1f}px")
        else:
            keep.append(True)
            last_valid_x = cx

    filtered = df[keep].copy()
    discarded = len(df) - len(filtered)
    print(f"[normalize] Kept {len(filtered)} / {len(df)} rows "
          f"({discarded} discarded)")
    return filtered

def chunk_mean_x(df, chunk_size=CHUNK_SIZE):
    """
    For every chunk of chunk_size rows, calculate the mean X.
    Adds a column 'chunk_mean_x' to the dataframe.
    """
    df = df.copy()
    df["chunk_id"]     = np.arange(len(df)) // chunk_size
    chunk_means        = df.groupby("chunk_id")["cx"].transform("mean")
    df["chunk_mean_x"] = chunk_means
    return df

def normalize_columns(df):
    """
    Normalize cx and cy to 0-1 range based on image size (128x128).
    Normalized values are what the NN will use.
    """
    IMAGE_SIZE = 128.0
    df         = df.copy()
    df["cx_norm"] = df["cx"] / IMAGE_SIZE
    df["cy_norm"] = df["cy"] / IMAGE_SIZE
    return df

def run():
    if not os.path.exists(LABELS_CSV):
        print(f"[normalize] labels.csv not found at {LABELS_CSV}")
        return

    df = pd.read_csv(LABELS_CSV)
    print(f"[normalize] Loaded {len(df)} rows from labels.csv")

    # Step 1 — calculate φ automatically
    phi = calculate_phi(df)

    # Step 2 — filter outliers based on X jumps
    df = filter_outliers(df, phi)

    # Step 3 — add chunk mean X every 150 rows
    df = chunk_mean_x(df, CHUNK_SIZE)

    # Step 4 — normalize cx and cy to 0-1
    df = normalize_columns(df)

    # Step 5 — save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[normalize] Saved → {OUTPUT_CSV}")
    print(f"[normalize] Columns: {list(df.columns)}")
    print(f"\n[normalize] Preview:")
    print(df[["frame_id", "cx", "cy", "chunk_mean_x",
              "cx_norm", "cy_norm"]].head(10).to_string(index=False))

if __name__ == "__main__":
    run()