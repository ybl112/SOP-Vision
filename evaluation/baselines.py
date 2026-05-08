"""
基线方法实现。

- Baseline 1：纯分类器，只评估工序步骤准确率，不检测违规。
"""

import numpy as np
from typing import Dict


def baseline1_classify_only(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Baseline 1：仅评估逐帧分类准确率，不具备违规检测能力。

    Args:
        y_true: 逐帧真实标签 (F,) 或 (N*T,)
        y_pred: 逐帧分类标签 (F,)

    Returns:
        {"accuracy": float, "violation_detection": "N/A"}
    """
    acc = float((y_true == y_pred).mean())
    return {"accuracy": acc, "violation_detection": "N/A"}
