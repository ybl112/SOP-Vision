"""
步骤3：端到端对比实验 — Proposed vs Baseline1 vs Baseline2。

用法:
    python scripts/3_run_experiment.py                          # 从 CSV 加载数据，用真实 ST-GCN 推理
    python scripts/3_run_experiment.py --skip_inference          # 使用预先保存的预测结果
"""

import sys
import os
import csv
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.config_loader import load_config
from src.knowledge.procedure_template import build_procedure_template
from src.knowledge.state_machine import ProcedureStateMachine
from src.alignment.utils import compress_sequence
from src.alignment.compliance_checker import (
    ProcedureComplianceChecker, method_two_baseline_dba,
)
from src.perception.st_gcn import STGCN
from evaluation.metrics import compute_violation_metrics
from evaluation.baselines import baseline1_classify_only


def _ranges_to_labels(ranges_str: str, total_frames: int) -> np.ndarray:
    """帧范围字符串 → 逐帧标签数组。格式: '0:0-12;1:13-32;...'"""
    labels = np.zeros(total_frames, dtype=int)
    for seg in ranges_str.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            label_str, frame_range = seg.split(":")
            label = int(label_str)
            s, e = frame_range.split("-")
            start, end = int(s), int(e)
            end = min(end + 1, total_frames)
            if start < total_frames:
                labels[start:end] = label
        except (ValueError, IndexError):
            continue
    return labels


def load_train_blocks(cfg: dict):
    """
    从训练集 CSV 和骨架 .npy 加载真实工序序列，供 Baseline2 DBA 模板构建。

    CSV 格式: filename,frame_ranges,step_order
    返回压缩后的工序块列表。
    """
    label_dir = cfg.get("dataset", {}).get("label_csv_dir", "data/labels/")
    skeleton_dir = cfg.get("dataset", {}).get("skeleton_dir", "data/skeletons/")
    train_csv = os.path.join(label_dir, "train.csv")

    if not os.path.exists(train_csv):
        print(f"  [WARN] 训练标注文件不存在: {train_csv}，Baseline2 将使用空模板")
        return []

    blocks_list = []
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            filename = row[0].strip()
            ranges_str = row[1].strip() if len(row) > 1 else ""
            if not ranges_str:
                continue

            npy_path = os.path.join(skeleton_dir, filename)
            if not os.path.exists(npy_path):
                continue
            total_frames = np.load(npy_path).shape[0]
            labels = _ranges_to_labels(ranges_str, total_frames)
            blocks = compress_sequence(labels.tolist())
            blocks_list.append(blocks)

    print(f"  加载 {len(blocks_list)} 个训练工序序列用于 DBA 模板")
    return blocks_list


# ========================= 真实数据加载 =========================

def load_test_data(cfg: dict):
    """
    从 CSV 标注和骨架 .npy 文件加载测试数据。

    CSV 格式: filename,frame_ranges,step_order
      例: sample_01.npy,0:0-5;1:6-22;2:23-39;4:40-56;5:57-70,1-2-4-5

    Returns:
        list of (filename, npy_path, label_seq, step_order_str)
    """
    label_dir = cfg.get("dataset", {}).get("label_csv_dir", "data/labels/")
    skeleton_dir = cfg.get("dataset", {}).get("skeleton_dir", "data/skeletons/")
    test_csv = os.path.join(label_dir, "test.csv")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"测试标注文件不存在: {test_csv}\n"
            "  格式: filename,frame_ranges,step_order  (帧范围如 0:0-5;1:6-22)"
        )

    samples = []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            filename = row[0].strip()
            ranges_str = row[1].strip() if len(row) > 1 else ""
            step_order = row[2].strip() if len(row) > 2 else ""

            npy_path = os.path.join(skeleton_dir, filename)
            if not os.path.exists(npy_path):
                print(f"  [WARN] 骨架文件不存在，跳过: {npy_path}")
                continue

            total_frames = np.load(npy_path).shape[0]
            labels = _ranges_to_labels(ranges_str, total_frames)
            samples.append((filename, npy_path, labels, step_order))

    print(f"  加载 {len(samples)} 个测试样本")
    return samples


def run_stgcn_inference(cfg: dict, samples: list, device: str = None):
    """
    用训练好的 ST-GCN 模型对测试样本逐帧推理。

    Args:
        cfg:     配置字典
        samples: load_test_data() 返回的样本列表
        device:  "cuda" / "cpu"

    Returns:
        list of (filename, pred_labels, gt_labels, step_order)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = cfg.get("train", {}).get("save_dir", "outputs/models/") + "best_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型权重不存在: {model_path}\n"
            "  请先运行 python scripts/2_train_classifier.py 训练模型"
        )

    num_classes = cfg.get("dataset", {}).get("num_classes", 7)
    window_size = cfg.get("model", {}).get("window_size", 32)

    model = STGCN(num_classes=num_classes, in_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    results = []
    for filename, npy_path, gt_labels, step_order in samples:
        data = np.load(npy_path).astype(np.float32)           # (T, 99)
        T_total = data.shape[0]
        data = data.reshape(T_total, 33, 3).transpose(1, 2, 0)  # (3, 33, T)
        data = data[np.newaxis, :, :, :]                         # (1, 3, 33, T)

        all_preds = []
        with torch.no_grad():
            for start in range(0, max(1, T_total - window_size + 1), window_size // 2):
                clip = data[:, :, :, start:start + window_size]
                if clip.shape[-1] < window_size:
                    continue
                tensor = torch.from_numpy(clip).to(device)       # (1, 3, 33, T)
                logits = model(tensor)                            # (1, C, T)
                preds = logits[0].argmax(dim=0).cpu().numpy()     # (T,)
                all_preds.append(preds)

        if not all_preds:
            pred_labels = np.zeros(len(gt_labels), dtype=int)
        else:
            pred_labels = np.concatenate(all_preds)[:len(gt_labels)]

        results.append((filename, pred_labels, gt_labels, step_order))

    return results


# ========================= 主实验 =========================

def run_experiment(skip_inference: bool = False):
    print("=" * 70)
    print("煤矿作业工序合规识别 — 对比实验")
    print("=" * 70)

    cfg = load_config()
    state_machine = ProcedureStateMachine()
    template = build_procedure_template(one_hot=False)

    # ---- 数据加载 ----
    print("\n[1/4] 加载测试数据 ...")
    test_samples = load_test_data(cfg)

    if skip_inference:
        print("\n[2/4] 跳过推理，使用预先保存的预测结果 ...")
        # 从 outputs/predictions/test_predictions.npy 加载
        pred_path = "outputs/predictions/test_predictions.npy"
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"预测文件不存在: {pred_path}")
        saved = np.load(pred_path, allow_pickle=True)
        pred_results = [(s[0], saved[i], s[2], s[3]) for i, s in enumerate(test_samples)]
    else:
        print("\n[2/4] ST-GCN 逐帧推理 ...")
        pred_results = run_stgcn_inference(cfg, test_samples)

    # ---- 对比实验 ----
    print("\n[3/4] 运行对比实验 ...")
    train_blocks = load_train_blocks(cfg)  # 从真实训练集构建 DBA 模板

    results = {
        "Proposed":  {"acc": [], "violations": [], "gt_violations": []},
        "Baseline1": {"acc": [], "violations": [], "gt_violations": []},
        "Baseline2": {"acc": [], "violations": [], "gt_violations": []},
    }

    for filename, preds, gts, step_order in pred_results:
        if len(gts) == 0:
            continue

        # 推断真实违规标注（从 step_order 解析）
        gt_violations = _parse_gt_violations(step_order, gts, state_machine)

        # Baseline1：纯逐帧分类
        b1 = baseline1_classify_only(gts, preds)
        results["Baseline1"]["acc"].append(b1["accuracy"])
        results["Baseline1"]["gt_violations"].extend(gt_violations)
        results["Baseline1"]["violations"].extend([])

        # Proposed：DTW + 状态机
        blocks = compress_sequence(preds.tolist())
        checker = ProcedureComplianceChecker(state_machine)
        proposed_v = checker.detect(blocks)
        acc_p = float((preds == gts).mean())
        results["Proposed"]["acc"].append(acc_p)
        results["Proposed"]["gt_violations"].extend(gt_violations)
        results["Proposed"]["violations"].extend(proposed_v)

        # Baseline2：DBA 模板 + DTW
        b2 = method_two_baseline_dba(
            train_blocks_list=train_blocks,
            test_blocks=blocks,
            test_probs=None,
            state_machine=state_machine,
        )
        acc_b2 = float((preds == gts).mean())
        results["Baseline2"]["acc"].append(acc_b2)
        results["Baseline2"]["gt_violations"].extend(gt_violations)
        results["Baseline2"]["violations"].extend(b2["violations"])

    # ---- 结果输出 ----
    print("\n[4/4] 评估结果 ...")
    _print_and_save_results(results)


def _parse_gt_violations(step_order: str, labels: np.ndarray,
                         state_machine: ProcedureStateMachine) -> list:
    """
    从 step_order 字符串解析真实违规标注。

    step_order 格式:
      "1-2-3-4-5"   → 标准流程，无违规
      "1-2-4-5"     → 漏步 (缺少步骤3)
      "1-3-2-4-5"   → 乱序
      "1-2-3"       → 非法终止
    """
    if not step_order:
        return []
    try:
        observed = [int(x) for x in step_order.split("-")]
    except ValueError:
        return []

    expected = list(range(1, state_machine.total_steps + 1))
    violations = []

    # 漏步检测
    missing = [s for s in expected if s not in observed]
    if missing:
        missing_names = [state_machine.step_names[s - 1] for s in missing]
        violations.append({
            "type": "漏步",
            "frame_start": len(labels) // 2,
            "missing_steps": missing_names,
            "current_step": state_machine.step_names[observed[-1] - 1],
            "description": f"缺少规程步骤: {missing_names}",
        })

    # 乱序检测
    filtered = [x for x in observed if x in expected]
    if filtered and filtered != sorted(filtered):
        violations.append({
            "type": "乱序",
            "frame_start": len(labels) // 2,
            "missing_steps": [],
            "current_step": state_machine.step_names[filtered[-1] - 1],
            "description": "步骤执行顺序与规程不符",
        })

    # 非法终止检测
    if observed[-1] != state_machine.end_state + 1:
        violations.append({
            "type": "非法终止",
            "frame_start": len(labels),
            "missing_steps": [],
            "current_step": state_machine.step_names[observed[-1] - 1],
            "description": f"工序在'{state_machine.step_names[observed[-1]-1]}'处结束，"
                           f"未到达终态'{state_machine.step_names[state_machine.end_state]}'",
        })

    return violations


def _print_and_save_results(results: dict):
    def eval_method(name, r):
        avg_acc = np.mean(r["acc"]) * 100 if r["acc"] else 0
        if name == "Baseline1":
            return {"accuracy": avg_acc, "漏步F1": "N/A", "乱序F1": "N/A", "非法终止F1": "N/A"}

        v_metrics = compute_violation_metrics(r["gt_violations"], r["violations"])
        return {
            "accuracy": avg_acc,
            "漏步F1": v_metrics["by_type"].get("漏步", {}).get("f1", 0),
            "乱序F1": v_metrics["by_type"].get("乱序", {}).get("f1", 0),
            "非法终止F1": v_metrics["by_type"].get("非法终止", {}).get("f1", 0),
        }

    proposed_eval = eval_method("Proposed", results["Proposed"])
    baseline1_eval = eval_method("Baseline1", results["Baseline1"])
    baseline2_eval = eval_method("Baseline2", results["Baseline2"])

    print()
    print(f"{'方法':<20} {'工序步骤准确率':<16} {'漏步F1':<10} {'乱序F1':<10} {'非法终止F1':<10}")
    print("-" * 66)
    print(f"{'纯分类器(Baseline1)':<20} {baseline1_eval['accuracy']:.1f}%{'':8} "
          f"{baseline1_eval['漏步F1']:<10} {baseline1_eval['乱序F1']:<10} "
          f"{baseline1_eval['非法终止F1']:<10}")
    print(f"{'DTW模板(Baseline2)':<20} {baseline2_eval['accuracy']:.1f}%{'':8} "
          f"{baseline2_eval['漏步F1']:.3f}{'':6} {baseline2_eval['乱序F1']:.3f}{'':6} "
          f"{baseline2_eval['非法终止F1']:.3f}")
    print(f"{'本文方法(Proposed)':<20} {proposed_eval['accuracy']:.1f}%{'':8} "
          f"{proposed_eval['漏步F1']:.3f}{'':6} {proposed_eval['乱序F1']:.3f}{'':6} "
          f"{proposed_eval['非法终止F1']:.3f}")

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["方法", "工序步骤准确率", "漏步F1", "乱序F1", "非法终止F1"])
        for name, ev in [("纯分类器(Baseline1)", baseline1_eval),
                          ("DTW模板(Baseline2)", baseline2_eval),
                          ("本文方法(Proposed)", proposed_eval)]:
            writer.writerow([name, f"{ev['accuracy']:.1f}%",
                             ev['漏步F1'], ev['乱序F1'], ev['非法终止F1']])

    print("\n结果已保存至 outputs/results/comparison.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SOP-Vision 对比实验")
    parser.add_argument("--skip_inference", action="store_true",
                        help="跳过 ST-GCN 推理，使用预先保存的 predictions")
    args = parser.parse_args()
    run_experiment(skip_inference=args.skip_inference)
