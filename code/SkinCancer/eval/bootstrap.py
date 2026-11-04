import numpy as np
import pandas as pd
from tqdm import tqdm

from SkinCancer.configs.training_config import BOOTSTRAP_N, SEED
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics


def bootstrap_macro_ci(y_true, y_prob, class_names, n_boot=BOOTSTRAP_N, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys_order = ["Accuracy", "Sensitivity", "Specificity", "AUC", "F1", "PPV", "NPV"]
    samples = {k: [] for k in keys_order}

    for _ in tqdm(range(n_boot), desc="Bootstrap 95% CI", leave=False):
        idx = rng.integers(0, n, size=n)
        try:
            overall, _ = calculate_detailed_multiclass_metrics(y_true[idx], y_prob[idx], class_names)
        except ValueError:
            continue
        for k in keys_order:
            samples[k].append(overall[k])

    rows = []
    for k in keys_order:
        vals = np.asarray(samples[k], dtype=float)
        rows.append(
            {
                "metric": k,
                "mean": float(np.mean(vals)),
                "ci_low": float(np.percentile(vals, 2.5)),
                "ci_high": float(np.percentile(vals, 97.5)),
                "n_boot_valid": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def print_macro_ci(ci_df):
    print("\n   [Overall Macro Metrics 95% CI]")
    for _, row in ci_df.iterrows():
        print(
            f"   {row['metric']:<30} | "
            f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )
