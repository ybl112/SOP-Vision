"""
DTW合规性违规检测器。

基于 fastdtw + 工序状态机，将观测到的工序序列与标准规程模板对齐，
检测漏步、乱序、非法终止三类违规。
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from src.knowledge.state_machine import ProcedureStateMachine
from src.knowledge.procedure_template import (
    build_procedure_template, class_id_to_step, get_step_name, CLASS_TO_STEP,
)
from src.alignment.utils import compress_sequence


# ------- DTW 核心 -------

class ProcedureComplianceChecker:
    """
    工序合规检查器。

    将压缩后的工序块序列与规程模板进行 DTW 对齐，检测违规。
    """

    def __init__(self, state_machine: ProcedureStateMachine = None,
                 window_size: int = 2, transition_cost: float = 0.1,
                 skip_penalty: float = 1.0):
        self.state_machine = state_machine or ProcedureStateMachine()
        self.window_size = window_size
        self.transition_cost = transition_cost
        self.skip_penalty = skip_penalty
        self.num_steps = self.state_machine.get_total_steps()

    # ---- 主检测接口 ----

    def detect(self, blocks: List[Dict],
               action_probs: Optional[np.ndarray] = None,
               return_path: bool = False):
        """
        检测工序违规。

        Args:
            blocks: 压缩后的工序块 [{"label":int, "start":int, "end":int}, ...]
            action_probs: 每个块的平均类别概率 (N, 7)，可选
            return_path: 是否同时返回 DTW 对齐路径（用于可视化）

        Returns:
            若 return_path=False: 违规列表
            若 return_path=True: (违规列表, 对齐路径, query数组, template数组)
        """
        violations = []
        filtered_blocks = [b for b in blocks if b["label"] in CLASS_TO_STEP]

        if len(filtered_blocks) < 2:
            return (violations, [], np.array([]), np.arange(self.num_steps)) if return_path else violations

        # 提取步骤序列
        step_seq = np.array([CLASS_TO_STEP[b["label"]] for b in filtered_blocks])

        # DTW 对齐  →  规程模板 (0,1,2,3,4)
        template = np.arange(self.num_steps, dtype=np.float64)
        query = step_seq.astype(np.float64)

        path = self._dtw_align(query, template)

        # 从对齐路径中检测违规
        violations += self._detect_from_path(path, query, template, filtered_blocks)
        violations += self._check_early_termination(step_seq, filtered_blocks)

        if return_path:
            return violations, path, query, template
        return violations

    # ---- DTW 对齐 ----

    def _dtw_align(self, query: np.ndarray, template: np.ndarray) -> List[tuple]:
        """
        DTW 对齐，使用自定义代价矩阵。

        Returns:
            path: [(q_idx, t_idx), ...] 最优对齐路径
        """
        try:
            from fastdtw import fastdtw

            dist, path = fastdtw(
                query.reshape(-1, 1),
                template.reshape(-1, 1),
                radius=min(self.window_size, max(len(query), len(template))),
                dist=self._custom_distance,
            )
            return path
        except ImportError:
            return self._simple_dtw(query, template)

    def _custom_distance(self, a, b):
        """
        自定义距离函数。

        - 完全匹配：0
        - 工序间调整匹配任何模板状态：0.1
        - 常规不匹配：1.0
        - 跳过模板状态：+1.0

        a, b 为 (1,) 数组，取标量值。
        """
        q_val = int(a[0])
        t_val = int(b[0])

        if q_val == t_val:
            return 0.0
        elif q_val == 6:  # 工序间调整
            return self.transition_cost
        else:
            return 1.0

    def _simple_dtw(self, query: np.ndarray, template: np.ndarray) -> List[tuple]:
        """纯 Python DTW 降级实现（避免 fastdtw 依赖缺失时崩溃）。"""
        n, m = len(query), len(template)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(max(1, i - self.window_size),
                           min(m + 1, i + self.window_size + 1)):
                cost = 0.0 if query[i - 1] == template[j - 1] else 1.0
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        # 回溯
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            candidates = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
            i, j = min(candidates, key=lambda x: dtw[x[0], x[1]])
        while i > 0:
            path.append((i - 1, 0))
            i -= 1
        while j > 0:
            path.append((0, j - 1))
            j -= 1
        return list(reversed(path))

    # ---- 违规检测 ----

    def _detect_from_path(self, path, query, template, blocks):
        violations = []
        prev_t = -1
        prev_q_val = -1

        for q_idx, t_idx in path:
            cur_q_val = int(query[q_idx])

            # ---- 漏步检测：查询序列中步骤跳跃 ----
            if prev_q_val >= 0 and cur_q_val > prev_q_val + 1:
                skipped = [get_step_name(s) for s in range(prev_q_val + 1, cur_q_val)]
                b = blocks[q_idx]
                violations.append({
                    "type": "漏步",
                    "frame_start": b["start"],
                    "missing_steps": skipped,
                    "current_step": get_step_name(cur_q_val),
                    "description": f"跳过了规程步骤: {skipped}，当前实际步骤: {get_step_name(cur_q_val)}",
                })

            # ---- 乱序检测：查询步骤回退 ----
            if prev_q_val >= 0 and cur_q_val < prev_q_val:
                b = blocks[q_idx]
                violations.append({
                    "type": "乱序",
                    "frame_start": b["start"],
                    "missing_steps": [],
                    "current_step": get_step_name(cur_q_val),
                    "description": f"步骤回退：从{get_step_name(prev_q_val)}回退到{get_step_name(cur_q_val)}",
                })

            prev_q_val = cur_q_val
            prev_t = t_idx

        return violations

    def _check_early_termination(self, step_seq, blocks):
        """检查是否非法终止（未到达终态即结束）。"""
        violations = []
        if len(step_seq) == 0:
            return violations

        end_state = self.state_machine.get_end_state()
        last_step = step_seq[-1]

        if last_step != end_state:
            end_name = self.state_machine.get_step_name(end_state)
            violations.append({
                "type": "非法终止",
                "frame_start": blocks[-1]["start"] if blocks else 0,
                "missing_steps": [],
                "current_step": get_step_name(last_step),
                "description": f"工序在'{get_step_name(last_step)}'处结束，未到达终态'{end_name}'",
            })

        return violations


def method_two_baseline_dba(train_blocks_list: List[List[Dict]],
                             test_blocks: List[Dict],
                             test_probs: Optional[np.ndarray],
                             state_machine: ProcedureStateMachine,
                             window_size: int = 2) -> Dict:
    """
    Baseline 2：使用 tslearn DBA 从多个演示中生成平均模板，再 DTW 对比。

    若 tslearn 不可用则降级为简单平均。
    """
    from src.alignment.utils import blocks_to_step_sequence

    # 收集所有训练序列
    sequences = []
    for blocks in train_blocks_list:
        steps = blocks_to_step_sequence(blocks, CLASS_TO_STEP)
        if len(steps) > 1:
            sequences.append(steps)

    if not sequences:
        return {"violations": [], "dtw_distance": float("inf"), "template": None}

    # DBA 平均模板
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance

        max_len = max(len(s) for s in sequences)
        ts_data = np.zeros((len(sequences), max_len))
        for i, s in enumerate(sequences):
            ts_data[i, :len(s)] = s

        avg_template = dtw_barycenter_averaging(ts_data, max_iter=10)
        template = avg_template.ravel()
    except ImportError:
        # 降级：简单多数投票
        max_len = max(len(s) for s in sequences)
        template = np.zeros(max_len)
        for i in range(max_len):
            votes = defaultdict(int)
            for s in sequences:
                if i < len(s):
                    votes[s[i]] += 1
            template[i] = max(votes, key=votes.get)

    template = template.astype(np.float64)

    # DTW 对齐
    test_steps = np.array(blocks_to_step_sequence(test_blocks, CLASS_TO_STEP), dtype=np.float64)
    checker = ProcedureComplianceChecker(state_machine, window_size=window_size)
    violations = checker.detect(test_blocks, test_probs)

    try:
        from fastdtw import fastdtw
        dtw_dist, path = fastdtw(
            test_steps.reshape(-1, 1),
            template.reshape(-1, 1),
            radius=min(window_size, max(len(test_steps), len(template))),
        )
    except ImportError:
        dtw_dist = float("inf")
        path = []

    return {"violations": violations, "dtw_distance": dtw_dist, "template": template.tolist()}
