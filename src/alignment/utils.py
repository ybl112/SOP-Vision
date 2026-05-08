"""序列压缩与DTW辅助工具。"""

import numpy as np
from typing import List, Dict


def compress_sequence(labels: List[int]) -> List[Dict]:
    """
    合并连续相同标签帧为工序块。

    Args:
        labels: 逐帧标签列表 list of int

    Returns:
        [{"label": int, "start": int, "end": int}, ...]
        end 为不包含边界，即区间 [start, end)
    """
    if not labels:
        return []

    blocks = []
    current_label = labels[0]
    start = 0

    for i, label in enumerate(labels):
        if label != current_label:
            blocks.append({"label": current_label, "start": start, "end": i})
            current_label = label
            start = i

    blocks.append({"label": current_label, "start": start, "end": len(labels)})
    return blocks


def blocks_to_sequence(blocks: List[Dict], length: int = None) -> np.ndarray:
    """从压缩块还原逐帧标签序列。"""
    if length is None:
        length = blocks[-1]["end"]
    seq = np.zeros(length, dtype=np.int64)
    for b in blocks:
        seq[b["start"]:b["end"]] = b["label"]
    return seq


def blocks_to_step_sequence(blocks: List[Dict], class_to_step: dict) -> List[int]:
    """将工序块转化为规程步骤索引序列（过滤非规程类）。"""
    steps = []
    for b in blocks:
        step = class_to_step.get(b["label"], -1)
        if step >= 0:
            steps.append(step)
    return steps


def build_onehot_block_features(blocks: List[Dict], num_classes: int = 5) -> np.ndarray:
    """
    将压缩块转为 one-hot 特征矩阵用于 DTW 对齐。
    仅保留规程类 (class_id 1-5)，映射到 5 维 one-hot。
    """
    mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

    features = []
    valid_blocks = []

    for b in blocks:
        if b["label"] in mapping:
            vec = np.zeros(num_classes, dtype=np.float32)
            vec[mapping[b["label"]]] = 1.0
            features.append(vec)
            valid_blocks.append(b)
        elif b["label"] == 6:  # 工序间调整
            vec = np.full(num_classes, 0.2, dtype=np.float32)  # 均匀低置信度
            features.append(vec)
            valid_blocks.append(b)
        # class_id=0 (开工检查) 忽略，其通常不在规程序列中

    return np.array(features, dtype=np.float32) if features else np.zeros((0, num_classes), dtype=np.float32)
