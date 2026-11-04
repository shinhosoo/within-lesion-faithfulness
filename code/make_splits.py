import argparse, os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ap = argparse.ArgumentParser()
ap.add_argument("--metadata", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

df = pd.read_csv(a.metadata).rename(columns={"dx": "diagnosis"})
df["diagnosis"] = df["diagnosis"].str.lower()
# dx_type, age, sex and localization are CC BY-NC metadata and are dropped here, so the
# regenerated splits carry only what this study redistributes.
keep = [c for c in ["lesion_id","image_id","diagnosis"] if c in df.columns]
df = df[keep]
print(f"total images={len(df)}, lesions={df.lesion_id.nunique()}")

def gsplit(d, size, seed):
    g = GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed)
    i, j = next(g.split(d, groups=d.lesion_id.values))
    return d.iloc[i].reset_index(drop=True), d.iloc[j].reset_index(drop=True)

tv, te = gsplit(df, 0.2, a.seed)
tr, va = gsplit(tv, 0.1/(1-0.2), a.seed)
parts = {"train": tr, "val": va, "test": te}

for x, y in [("train","val"),("train","test"),("val","test")]:
    ov = set(parts[x].lesion_id) & set(parts[y].lesion_id)
    assert not ov, f"lesion leakage {x}/{y}: {len(ov)}"
print("lesion-level leakage check passed")

for n, d in parts.items():
    d = d.sort_values(["lesion_id","image_id"]).reset_index(drop=True)
    d.to_csv(f"{a.out}/global_{n}.csv", index=False)
    print(f"{n:5s} {len(d):5d} images  {d.lesion_id.nunique():5d} lesions")

P = {"nv":1336,"mel":236,"bkl":243,"df":18,"vasc":34,"bcc":90,"akiec":67}
c = parts["test"].diagnosis.value_counts()
got = {k: int(c.get(k,0)) for k in P}
print("test support:", got)
print("test split:", "MATCH" if got == P else "MISMATCH")
