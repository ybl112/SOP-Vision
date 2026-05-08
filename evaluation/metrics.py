"""评估指标：准确率、精确率、召回率、F1。"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """逐帧分类准确率。"""
    return float(accuracy_score(y_true, y_pred))


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: List[str] = None,
                              average: str = "macro") -> Dict:
    """
    计算精确率、召回率、F1。

    Args:
        y_true: 真实标签 (N,)
        y_pred: 分类标签 (N,)
        class_names: 类别名称列表
        average: "macro" / "micro" / "weighted"

    Returns:
        {"precision": float, "recall": float, "f1": float,
         "per_class": {name: {"precision":, "recall":, "f1":}, ...}}
    """
    unique = np.unique(np.concatenate([y_true, y_pred]))

    precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))

    result = {"precision": precision, "recall": recall, "f1": f1, "per_class": {}}

    if class_names is None:
        class_names = [f"class_{i}" for i in unique]

    for i, name in enumerate(class_names):
        if i in unique:
            p = float(precision_score(y_true == i, y_pred == i, zero_division=0))
            r = float(recall_score(y_true == i, y_pred == i, zero_division=0))
            f = float(f1_score(y_true == i, y_pred == i, zero_division=0))
        else:
            p = r = f = 0.0
        result["per_class"][name] = {"precision": p, "recall": r, "f1": f}

    return result


def compute_violation_metrics(gt_violations: List[Dict],
                               pred_violations: List[Dict],
                               violation_types: List[str] = None) -> Dict:
    """
    违规检测评估（按类型分别计算精确率/召回率/F1）。

    匹配规则：type 相同且 frame_start 误差在 ±15 帧以内视为命中。

    Returns:
        {"overall": {"precision","recall","f1"},
         "by_type": {"漏步": {...}, "乱序": {...}, "非法终止": {...}}}
    """
    if violation_types is None:
        violation_types = ["漏步", "乱序", "非法终止"]

    def _match(gt_list, pred_list):
        matched = 0
        pred_matched = set()
        for g in gt_list:
            for pi, p in enumerate(pred_list):
                if pi in pred_matched:
                    continue
                if g["type"] == p["type"] and abs(g.get("frame_start", 0) - p.get("frame_start", 0)) <= 15:
                    matched += 1
                    pred_matched.add(pi)
                    break
        return matched

    total_gt = len(gt_violations)
    total_pred = len(pred_violations)
    matched = _match(gt_violations, pred_violations)

    tp = matched
    fp = max(0, total_pred - matched)
    fn = max(0, total_gt - matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {"overall": {"precision": precision, "recall": recall, "f1": f1}, "by_type": {}}

    for vtype in violation_types:
        gt_sub = [v for v in gt_violations if v["type"] == vtype]
        pred_sub = [v for v in pred_violations if v["type"] == vtype]
        matched_sub = _match(gt_sub, pred_sub)

        tp_s = matched_sub
        fp_s = max(0, len(pred_sub) - matched_sub)
        fn_s = max(0, len(gt_sub) - matched_sub)

        p = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
        r = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        result["by_type"][vtype] = {"precision": p, "recall": r, "f1": f}

    return result
