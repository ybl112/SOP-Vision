"""
合规模板构建模块。

将 YAML 规程定义中的 class_id 映射到工序步骤索引，
并为 DTW 对齐提供标准规程模板。
"""

import numpy as np
from typing import Dict, Optional

from src.knowledge.state_machine import ProcedureStateMachine

# 全局单例，首次调用时从 YAML 加载
_statemachine: Optional[ProcedureStateMachine] = None

# ---- 类别 → 步骤 映射 ----

# CLASS_TO_STEP: class_id (int) → step_index (int)
# 由 state_machine 的 step_class_map 反向构建
CLASS_TO_STEP: Dict[int, int] = {}

# STEP_NAMES: step_index → 步骤中文名
STEP_NAMES: Dict[int, str] = {}

# CLASS_NAMES: class_id → 类别中文名
CLASS_NAMES: Dict[int, str] = {}


def _ensure_loaded():
    """延迟加载规程定义，避免循环导入。"""
    global _statemachine, CLASS_TO_STEP, STEP_NAMES, CLASS_NAMES
    if _statemachine is not None:
        return
    _statemachine = ProcedureStateMachine()
    # class_id → step_index
    CLASS_TO_STEP = {s["class_id"]: s["index"] for s in _statemachine.steps}
    # step_index → name
    STEP_NAMES = {s["index"]: s["name"] for s in _statemachine.steps}
    # class_id → name
    CLASS_NAMES = {s["class_id"]: s["name"] for s in _statemachine.steps}


# ---- 对外 API ----

def class_id_to_step(class_id: int) -> int:
    """将动作类别 ID 映射为规程步骤索引。"""
    _ensure_loaded()
    return CLASS_TO_STEP.get(class_id, -1)


def get_step_name(step_idx: int) -> str:
    """根据步骤索引返回中文步骤名。"""
    _ensure_loaded()
    return STEP_NAMES.get(step_idx, f"未知步骤({step_idx})")


def get_class_name(class_id: int) -> str:
    """根据类别 ID 返回中文类别名。"""
    _ensure_loaded()
    return CLASS_NAMES.get(class_id, f"未知类别({class_id})")


def build_procedure_template(one_hot: bool = False) -> np.ndarray:
    """
    构建标准规程模板。

    Args:
        one_hot: True 时返回 (N, N) 单位矩阵，供 DTW 对齐使用。

    Returns:
        one_hot=False: (5,)  array [0,1,2,3,4]
        one_hot=True:  (5,5) identity matrix
    """
    _ensure_loaded()
    n = _statemachine.total_steps
    if one_hot:
        return np.eye(n, dtype=np.float64)
    return np.arange(n, dtype=np.float64)


# 模块加载时立即初始化，确保 CLASS_TO_STEP 等全局变量可用
_ensure_loaded()
