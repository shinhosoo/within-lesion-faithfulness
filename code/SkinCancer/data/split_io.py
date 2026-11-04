import glob
import os

import pandas as pd

from SkinCancer.configs.data_config import (
    DATA_ROOT,
    IMAGE_DIRS,
    MEL_CLASS_NAMES,
    METADATA_CSV,
    SPLIT_ROOT,
    FINAL7_TEST_CSV,
    NM_CLASS_NAMES,
)
from SkinCancer.configs.data_config import STAGE2_FINAL_MAPPING
from SkinCancer.utils.logging import debug_dataframe


def build_image_path_map(image_dirs=IMAGE_DIRS):
    path_map = {}
    print("\n[Debug][DataPath]")
    print(f"  DATA_ROOT={DATA_ROOT}")
    print(f"  METADATA_CSV={METADATA_CSV}")
    print(f"  SPLIT_ROOT={SPLIT_ROOT}")
    for d in image_dirs:
        print(f"  image_dir={d}")
        if not os.path.isdir(d):
            print(f"[Warning] Directory not found: {d}")
            continue
        jpg_paths = glob.glob(os.path.join(d, "*.jpg"))
        print(f"  image_count[{d}]={len(jpg_paths)}")
        for fpath in jpg_paths:
            image_id = os.path.splitext(os.path.basename(fpath))[0]
            path_map[image_id] = fpath
    print(f"[Data] Found {len(path_map)} images across {len(image_dirs)} directories.")
    return path_map


def load_metadata(path_map):
    df = pd.read_csv(METADATA_CSV)
    df = df.rename(columns={"dx": "diagnosis", "image_id": "image_id"})
    df["image_path"] = df["image_id"].map(path_map)
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"[Warning] {missing} records have no matching image and will be dropped.")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    print(f"[Data] Metadata loaded: {len(df)} valid samples.")
    debug_dataframe("metadata", df)
    return df


def _load_fixed_split(csv_path, path_map, label_mapping, label_col, allowed_dx=None):
    print(f"\n[Debug][Split] csv_path={csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[Debug][Split] raw_rows={len(df)}")

    if "diagnosis" not in df.columns and "dx" in df.columns:
        df = df.rename(columns={"dx": "diagnosis"})

    if allowed_dx is not None:
        df = df[df["diagnosis"].isin(allowed_dx)].copy()
        print(f"[Debug][Split] rows_after_allowed_dx_filter={len(df)} allowed_dx={allowed_dx}")

    df[label_col] = df["diagnosis"].map(label_mapping)

    if "image_path" not in df.columns:
        df["image_path"] = df["image_id"].map(path_map)

    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"[Warning] {missing} records in {csv_path} have no matching image and will be dropped.")

    df = df.dropna(subset=[label_col, "image_path"]).reset_index(drop=True)
    debug_dataframe(os.path.basename(csv_path), df, label_col=label_col)
    return df


def load_final7_test_data(path_map):
    """Load the test split used for the 7-class evaluation."""
    df = _load_fixed_split(FINAL7_TEST_CSV, path_map, STAGE2_FINAL_MAPPING, "final_label")
    print(f"[Debug][Final7-Test] test_samples={len(df)}")
    return df
