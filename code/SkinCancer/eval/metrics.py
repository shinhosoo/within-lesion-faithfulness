import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def calculate_binary_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "PPV": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
    }


def calculate_detailed_binary_metrics(y_true, y_pred, y_prob, class_names):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    metrics = calculate_binary_metrics(y_true, y_pred, y_prob)
    per_class = {}

    for class_idx, cls in enumerate(class_names):
        y_true_bin = (y_true == class_idx).astype(int)
        y_pred_bin = (y_pred == class_idx).astype(int)
        y_prob_bin = y_prob if class_idx == 1 else 1.0 - y_prob
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        per_class[cls] = {
            "Support": int(y_true_bin.sum()),
            "Predicted": int(y_pred_bin.sum()),
            "Accuracy": accuracy_score(y_true_bin, y_pred_bin),
            "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "AUC": roc_auc_score(y_true_bin, y_prob_bin) if len(np.unique(y_true_bin)) > 1 else 0.5,
            "F1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
            "PPV": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }
    return metrics, per_class


def calculate_detailed_multiclass_metrics(y_true, y_prob, class_names):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.argmax(y_prob, axis=1)
    acc_overall = accuracy_score(y_true, y_pred)
    per_class = {}
    sens_list, spec_list, auc_list, f1_list, ppv_list, npv_list = [], [], [], [], [], []

    for i, cls in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        y_prob_bin = y_prob[:, i]
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        if len(np.unique(y_true_bin)) == 2:
            auc_cls = roc_auc_score(y_true_bin, y_prob_bin)
            auc_list.append(auc_cls)
        else:
            auc_cls = 0.5

        per_class[cls] = {
            "Support": int(y_true_bin.sum()),
            "Predicted": int(y_pred_bin.sum()),
            "Accuracy": accuracy_score(y_true_bin, y_pred_bin),
            "Sensitivity": sens,
            "Specificity": spec,
            "AUC": auc_cls,
            "F1": f1,
            "PPV": ppv,
            "NPV": npv,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }

        if (tp + fn) > 0 or (tp + fp) > 0:
            sens_list.append(sens)
            spec_list.append(spec)
            f1_list.append(f1)
            ppv_list.append(ppv)
            npv_list.append(npv)

    overall = {
        "Accuracy": acc_overall,
        "Sensitivity": np.mean(sens_list) if len(sens_list) > 0 else 0.0,
        "Specificity": np.mean(spec_list) if len(spec_list) > 0 else 0.0,
        "AUC": np.mean(auc_list) if len(auc_list) > 0 else 0.5,
        "F1": np.mean(f1_list) if len(f1_list) > 0 else 0.0,
        "PPV": np.mean(ppv_list) if len(ppv_list) > 0 else 0.0,
        "NPV": np.mean(npv_list) if len(npv_list) > 0 else 0.0,
    }
    return overall, per_class


def print_metrics(metrics, stage_name):
    print(f"\n{'=' * 70}\n   {stage_name}\n{'=' * 70}")
    for k, v in metrics.items():
        print(f"   {k:<30} | {v:.4f}")


def print_detailed_binary_metrics(overall, per_class, stage_name):
    print_metrics(overall, stage_name)
    _print_per_class_metrics(per_class, class_width=18)


def print_detailed_metrics(overall, per_class, stage_name):
    print(f"\n{'=' * 70}\n   {stage_name}\n{'=' * 70}")
    print("   [Overall Macro Metrics]")
    for k in ["Accuracy", "Sensitivity", "Specificity", "AUC", "F1", "PPV", "NPV"]:
        print(f"   {k:<30} | {overall[k]:.4f}")
    _print_per_class_metrics(per_class, class_width=10)


def _print_per_class_metrics(per_class, class_width):
    print("\n   [Per-Class Metrics (One-vs-Rest)]")
    header = (
        f"   {'Class':<{class_width}} | {'Support':>7} | {'Pred':>7} | {'Acc':>6} | "
        f"{'Sens':>6} | {'Spec':>6} | {'AUC':>6} | {'F1':>6} | "
        f"{'PPV':>6} | {'NPV':>6} | {'TN':>5} | {'FP':>5} | {'FN':>5} | {'TP':>5}"
    )
    print(header)
    print("   " + "-" * (106 + class_width))
    for cls, m in per_class.items():
        print(
            f"   {cls:<{class_width}} | {m['Support']:>7} | {m['Predicted']:>7} | "
            f"{m['Accuracy']:>6.4f} | {m['Sensitivity']:>6.4f} | "
            f"{m['Specificity']:>6.4f} | {m['AUC']:>6.4f} | "
            f"{m['F1']:>6.4f} | {m['PPV']:>6.4f} | {m['NPV']:>6.4f} | "
            f"{m['TN']:>5} | {m['FP']:>5} | {m['FN']:>5} | {m['TP']:>5}"
        )

