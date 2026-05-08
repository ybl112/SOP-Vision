"""
基线方法实现。

- Baseline 1：纯分类器，只评估工序步骤准确率，不检测违规。
- Baseline 2：DBA 平均模板 + DTW 对齐，进行违规检测。
"""

import numpy as np
from typing import Dict, List

from src.alignment.compliance_checker import method_two_baseline_dba
from src.knowledge.state_machine import ProcedureStateMachine
from src.alignment.utils import compress_sequence


def baseline1_classify_only(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Baseline 1：仅评估逐帧分类准确率，不具备违规检测能力。

    Args:
        y_true: 逐帧真实标签 (F,) 或 (N*T,)
        y_pred: 逐帧预测标签 (F,)

    Returns:
        {"accuracy": float, "violation_detection": "N/A"}
    """
    acc = float((y_true == y_pred).mean())
    return {"accuracy": acc, "violation_detection": "N/A"}


def baseline2_dtw_template(train_blocks_list: List[List[Dict]],
                            test_labels: List[int],
                            state_machine: ProcedureStateMachine = None,
                            window_size: int = 2) -> Dict:
    """
    Baseline 2：对多个训练演示进行 DBA 平均，再用 DTW 做违规检测。

    Args:
        train_blocks_list: 训练集压缩块列表（多个样本）
        test_labels:       测试集逐帧标签
        state_machine:     工序状态机
        window_size:       Sakoe-Chiba 窗口

    Returns:
        {"violations": [...], "dtw_distance": float}
    """
    if state_machine is None:
        state_machine = ProcedureStateMachine()

    test_blocks = compress_sequence(test_labels)

    return method_two_baseline_dba(
        train_blocks_list=train_blocks_list,
        test_blocks=test_blocks,
        test_probs=None,
        state_machine=state_machine,
        window_size=window_size,
    )
