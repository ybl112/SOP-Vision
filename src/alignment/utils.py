"""序列压缩与DTW辅助工具。"""

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


def blocks_to_step_sequence(blocks: List[Dict], class_to_step: dict) -> List[int]:
    """将工序块转化为规程步骤索引序列（过滤非规程类）。"""
    steps = []
    for b in blocks:
        step = class_to_step.get(b["label"], -1)
        if step >= 0:
            steps.append(step)
    return steps
