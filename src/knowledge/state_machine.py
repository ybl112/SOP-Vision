"""工序状态机 — 基于 YAML 规程定义的合法状态转移检查。"""

import os
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional


class ProcedureStateMachine:
    """
    锚杆支护作业规程状态机。

    从 YAML 规程文件读取步骤定义与合法转移矩阵，提供：
    - check_transition: 判断状态转移是否合法
    - get_step_name / get_total_steps: 基本信息查询
    """

    def __init__(self, rules_path: str = None):
        if rules_path is None:
            rules_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "config", "procedure_rules", "switching_operation.yaml",
            )
        self.rules_path = rules_path
        self._load_rules(rules_path)

    def _load_rules(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"规程文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.procedure_name = data["procedure_name"]
        self.total_steps = data["total_steps"]
        self.start_state = data.get("start_state", 0)
        self.end_state = data.get("end_state", self.total_steps - 1)

        # 步骤信息
        self.steps: List[Dict] = data["steps"]
        self.step_names: List[str] = [s["name"] for s in self.steps]
        self.step_class_map: Dict[int, int] = {s["index"]: s["class_id"] for s in self.steps}

        # 构建 N×N 合法转移矩阵
        self.transition_matrix = np.zeros((self.total_steps, self.total_steps), dtype=bool)
        for src, dst in data.get("valid_transitions", []):
            if 0 <= src < self.total_steps and 0 <= dst < self.total_steps:
                self.transition_matrix[src, dst] = True

    # ---- API ----

    def check_transition(self, current_state: int, next_state: int) -> Dict:
        """
        检查状态转移的合法性。

        Args:
            current_state: 当前规程步骤索引 (0-4)，允许 -1 表示尚未进入规程
            next_state:    目标规程步骤索引 (0-4)，允许 -1 表示不涉及规程步骤

        Returns:
            {"valid": bool, "violation_type": "漏步"|"乱序"|"非法转移"|None, "detail": str}
        """
        # 若 next_state 为非规程类（开工检查/工序间调整），不检查转移
        if next_state == -1:
            return {"valid": True, "violation_type": None, "detail": ""}

        # 尚未进入规程：首次进入必须是 step 0
        if current_state == -1:
            if next_state == 0:
                return {"valid": True, "violation_type": None, "detail": ""}
            else:
                return {
                    "valid": False,
                    "violation_type": "漏步",
                    "detail": f"首次进入规程应为{self.step_names[0]}，实际为{self.step_names[next_state]}",
                }

        # 正常转移检查
        if self.transition_matrix[current_state, next_state]:
            return {"valid": True, "violation_type": None, "detail": ""}

        # 非法转移分类
        if next_state > current_state + 1:
            skipped = [self.step_names[i] for i in range(current_state + 1, next_state)]
            return {
                "valid": False,
                "violation_type": "漏步",
                "detail": f"跳过步骤: {skipped}",
            }
        elif next_state < current_state:
            return {
                "valid": False,
                "violation_type": "乱序",
                "detail": f"从{self.step_names[current_state]}回退到{self.step_names[next_state]}",
            }
        else:
            return {
                "valid": False,
                "violation_type": "非法转移",
                "detail": f"{self.step_names[current_state]} → {self.step_names[next_state]}",
            }

    def get_step_name(self, state_idx: int) -> str:
        if 0 <= state_idx < len(self.step_names):
            return self.step_names[state_idx]
        return f"未知步骤({state_idx})"

    def get_total_steps(self) -> int:
        return self.total_steps

    def get_start_state(self) -> int:
        return self.start_state

    def get_end_state(self) -> int:
        return self.end_state
