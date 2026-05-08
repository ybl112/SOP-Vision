"""
步骤3：端到端对比实验 — Proposed vs Baseline1 vs Baseline2。

用法:
    python scripts/3_run_experiment.py                          # 从骨架 + 模型推理
    python scripts/3_run_experiment.py --skip_inference          # 使用缓存的推理结果
"""

import sys
import os
import csv
import argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from src.config_loader import load_config
from src.knowledge.procedure_template import build_procedure_template
from src.knowledge.state_machine import ProcedureStateMachine
from src.alignment.utils import compress_sequence
from src.alignment.compliance_checker import (
    ProcedureComplianceChecker, method_two_baseline_dba,
)
from src.perception.st_gcn import STGCN
from src.data.dataset import _parse_frame_ranges as _ranges_to_labels
from evaluation.metrics import compute_violation_metrics
from evaluation.baselines import baseline1_classify_only

# 中文字体
_cjk = [f.name for f in fm.fontManager.ttflist
        if any(k in f.name for k in ("SimHei", "Microsoft YaHei", "SimSun", "Noto Sans CJK"))]
if _cjk:
    plt.rcParams["font.sans-serif"] = [_cjk[0]] + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# ========================= 数据加载 =========================

def load_train_blocks(cfg: dict):
    """从训练集 CSV 和骨架 .npy 加载工序序列，供 Baseline2 DBA 模板构建。"""
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


def load_test_data(cfg: dict):
    """从 CSV 标注和骨架 .npy 加载测试数据。"""
    label_dir = cfg.get("dataset", {}).get("label_csv_dir", "data/labels/")
    skeleton_dir = cfg.get("dataset", {}).get("skeleton_dir", "data/skeletons/")
    test_csv = os.path.join(label_dir, "test.csv")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"测试标注文件不存在: {test_csv}\n"
            "  格式: filename,frame_ranges,step_order"
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


# ========================= ST-GCN 推理 =========================

def run_stgcn_inference(cfg: dict, samples: list, device: str = None):
    """用训练好的 ST-GCN 模型对测试样本逐帧推理，结果缓存到 outputs/inference/。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = cfg.get("train", {}).get("save_dir", "outputs/models/")
    model_path = os.path.join(save_dir, "best_model.pt")
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
        data = np.load(npy_path).astype(np.float32)
        T_total = data.shape[0]
        data = data.reshape(T_total, 33, 3).transpose(1, 2, 0)
        data = data[np.newaxis, :, :, :]

        all_preds = []
        with torch.no_grad():
            for start in range(0, max(1, T_total - window_size + 1), window_size // 2):
                clip = data[:, :, :, start:start + window_size]
                if clip.shape[-1] < window_size:
                    continue
                tensor = torch.from_numpy(clip).to(device)
                logits = model(tensor)
                preds = logits[0].argmax(dim=0).cpu().numpy()
                all_preds.append(preds)

        if not all_preds:
            pred_labels = np.zeros(len(gt_labels), dtype=int)
        else:
            pred_labels = np.concatenate(all_preds)[:len(gt_labels)]

        results.append((filename, pred_labels, gt_labels, step_order))

    # 缓存推理结果
    cache_dir = cfg.get("output", {}).get("inference_dir", "outputs/inference/")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "test_inference.npy"),
            np.array([p for _, p, _, _ in results], dtype=object))
    print(f"  推理缓存已保存至 {cache_dir}")

    return results


# ========================= 主实验 =========================

def run_experiment(skip_inference: bool = False):
    print("=" * 70)
    print("煤矿作业工序合规识别 — 对比实验")
    print("=" * 70)

    cfg = load_config()
    state_machine = ProcedureStateMachine()
    template = build_procedure_template(one_hot=False)

    compliance_cfg = cfg.get("compliance", {})
    checker_kwargs = {
        "window_size": compliance_cfg.get("window_size", 2),
        "transition_cost": compliance_cfg.get("transition_cost", 0.1),
        "skip_penalty": compliance_cfg.get("skip_penalty", 1.0),
    }

    out_cfg = cfg.get("output", {})
    result_dir = out_cfg.get("result_dir", "outputs/results/")
    dtw_dir = out_cfg.get("dtw_dir", "outputs/dtw/")
    timeline_dir = out_cfg.get("timeline_dir", "outputs/timeline/")
    cache_dir = out_cfg.get("inference_dir", "outputs/inference/")

    # ---- 数据加载 ----
    print("\n[1/4] 加载测试数据 ...")
    test_samples = load_test_data(cfg)

    if skip_inference:
        print("\n[2/4] 跳过推理，使用缓存结果 ...")
        cache_path = os.path.join(cache_dir, "test_inference.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"推理缓存不存在: {cache_path}\n"
                "  请先不带 --skip_inference 运行一次以生成缓存"
            )
        saved = np.load(cache_path, allow_pickle=True)
        if len(saved) != len(test_samples):
            raise ValueError(
                f"缓存样本数 ({len(saved)}) 与测试样本数 ({len(test_samples)}) 不匹配"
            )
        pred_results = [(s[0], saved[i], s[2], s[3]) for i, s in enumerate(test_samples)]
    else:
        print("\n[2/4] ST-GCN 逐帧推理 ...")
        pred_results = run_stgcn_inference(cfg, test_samples)

    # ---- 对比实验 ----
    print("\n[3/4] 运行对比实验 ...")
    train_blocks = load_train_blocks(cfg)

    results = {
        "Proposed":  {"acc": [], "violations": [], "gt_violations": []},
        "Baseline1": {"acc": [], "violations": [], "gt_violations": []},
        "Baseline2": {"acc": [], "violations": [], "gt_violations": []},
    }

    for filename, preds, gts, step_order in pred_results:
        if len(gts) == 0:
            continue

        gt_violations = _parse_gt_violations(step_order, gts, state_machine)

        # Baseline1
        b1 = baseline1_classify_only(gts, preds)
        results["Baseline1"]["acc"].append(b1["accuracy"])
        results["Baseline1"]["gt_violations"].extend(gt_violations)
        results["Baseline1"]["violations"].extend([])

        # Proposed: DTW + 状态机
        blocks = compress_sequence(preds.tolist())
        checker = ProcedureComplianceChecker(state_machine, **checker_kwargs)
        proposed_v, dtw_path, query, tpl = checker.detect(blocks, return_path=True)
        acc_p = float((preds == gts).mean())
        results["Proposed"]["acc"].append(acc_p)
        results["Proposed"]["gt_violations"].extend(gt_violations)
        results["Proposed"]["violations"].extend(proposed_v)

        # Baseline2: DBA 模板 + DTW
        b2 = method_two_baseline_dba(
            train_blocks_list=train_blocks, test_blocks=blocks,
            test_probs=None, state_machine=state_machine,
            window_size=checker_kwargs["window_size"],
        )
        acc_b2 = float((preds == gts).mean())
        results["Baseline2"]["acc"].append(acc_b2)
        results["Baseline2"]["gt_violations"].extend(gt_violations)
        results["Baseline2"]["violations"].extend(b2["violations"])

        # 推理时直接出图（和 YOLO 一样的行为）
        _save_dtw_plot(query, tpl, dtw_path, dtw_dir, filename)
        _save_timeline_plot(blocks, proposed_v, gts, preds, timeline_dir, filename)

    # ---- 结果输出 ----
    print("\n[4/4] 评估结果 ...")
    _print_and_save_results(results, result_dir)


# ========================= 违规标注解析 =========================

def _parse_gt_violations(step_order: str, labels: np.ndarray,
                         state_machine: ProcedureStateMachine) -> list:
    """从 step_order 字符串解析真实违规标注。"""
    if not step_order:
        return []
    try:
        observed = [int(x) for x in step_order.split("-")]
    except ValueError:
        return []

    expected = list(range(1, state_machine.total_steps + 1))
    violations = []

    missing = [s for s in expected if s not in observed]
    if missing:
        missing_names = [state_machine.step_names[s - 1] for s in missing]
        violations.append({
            "type": "漏步", "frame_start": len(labels) // 2,
            "missing_steps": missing_names,
            "current_step": state_machine.step_names[observed[-1] - 1],
            "description": f"缺少规程步骤: {missing_names}",
        })

    filtered = [x for x in observed if x in expected]
    if filtered and filtered != sorted(filtered):
        violations.append({
            "type": "乱序", "frame_start": len(labels) // 2,
            "missing_steps": [],
            "current_step": state_machine.step_names[filtered[-1] - 1],
            "description": "步骤执行顺序与规程不符",
        })

    if observed[-1] != state_machine.end_state + 1:
        violations.append({
            "type": "非法终止", "frame_start": len(labels),
            "missing_steps": [],
            "current_step": state_machine.step_names[observed[-1] - 1],
            "description": (
                f"工序在'{state_machine.step_names[observed[-1]-1]}'处结束，"
                f"未到达终态'{state_machine.step_names[state_machine.end_state]}'"
            ),
        })

    return violations


# ========================= 结果输出 =========================

def _print_and_save_results(results: dict, result_dir: str = "outputs/results/"):
    """打印评估结果并保存 CSV。"""

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

    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["方法", "工序步骤准确率", "漏步F1", "乱序F1", "非法终止F1"])
        for name, ev in [("纯分类器(Baseline1)", baseline1_eval),
                          ("DTW模板(Baseline2)", baseline2_eval),
                          ("本文方法(Proposed)", proposed_eval)]:
            writer.writerow([name, f"{ev['accuracy']:.1f}%",
                             ev['漏步F1'], ev['乱序F1'], ev['非法终止F1']])

    print(f"\n结果已保存至 {csv_path}")


# ========================= 内联可视化 =========================

def _save_dtw_plot(query: np.ndarray, template: np.ndarray, path: list,
                   save_dir: str, filename: str = ""):
    """DTW 代价矩阵 + 对齐路径（每个样本一张图）。"""
    if len(query) == 0 or len(template) == 0:
        return

    n, m = len(query), len(template)
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = 0.0 if query[i] == template[j] else 1.0

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cost, origin="lower", cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    if path:
        px, py = zip(*path)
        ax.plot(py, px, "b-", linewidth=1.5, alpha=0.8, label="DTW对齐路径")

    ax.set_xlabel("规程模板步骤", fontsize=10)
    ax.set_ylabel("观测步骤序列", fontsize=10)
    ax.set_xticks(range(m))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"S{i}" for i in template.astype(int)])
    ax.set_yticklabels([f"S{i}" for i in query.astype(int)])

    title = f"DTW 对齐 — {filename}" if filename else "DTW 对齐"
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8, label="不匹配代价")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    safe = filename.replace("/", "_").replace("\\", "_") or "sample"
    fig.savefig(os.path.join(save_dir, f"{safe}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_timeline_plot(blocks: list, violations: list,
                        gt_labels: np.ndarray, pred_labels: np.ndarray,
                        save_dir: str, filename: str = ""):
    """违规检测时间线图（每个样本一张图）。"""
    total_frames = len(gt_labels)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    # 上轴：真实/预测标签
    ax0 = axes[0]
    ax0.step(range(total_frames), gt_labels, where="post", linewidth=0.8,
             color="#1f77b4", alpha=0.7, label="真实标签")
    ax0.step(range(total_frames), pred_labels, where="post", linewidth=0.8,
             color="#ff7f0e", alpha=0.7, label="预测标签")
    ax0.set_ylabel("动作类别", fontsize=9)
    ax0.set_ylim(-0.5, 6.5)
    ax0.legend(loc="upper right", fontsize=8, ncol=2)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # 下轴：工序块 + 违规标注
    ax1 = axes[1]
    colors = ["#d9d9d9", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"]
    for b in blocks:
        label = b["label"]
        color = colors[label] if 0 < label <= 5 else "#eeeeee"
        ax1.axvspan(b["start"], b["end"], alpha=0.5, color=color, edgecolor="none")
        mid = (b["start"] + b["end"]) / 2
        ax1.text(mid, 0.5, str(label), ha="center", va="center", fontsize=7,
                 color="#333333", fontweight="bold")

    v_colors = {"漏步": "#e41a1c", "乱序": "#ff7f00", "非法终止": "#984ea3"}
    for v in violations:
        vtype = v.get("type", "?")
        fs = v.get("frame_start", 0)
        color = v_colors.get(vtype, "#000000")
        ax1.axvline(x=fs, color=color, linewidth=2, linestyle="--", alpha=0.8)
        ax1.annotate(vtype, xy=(fs, 0.85), fontsize=8, color=color,
                     rotation=90, va="bottom", fontweight="bold")

    ax1.set_ylabel("工序块 + 违规", fontsize=9)
    ax1.set_xlabel("帧序号", fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)

    title = f"违规检测 — {filename}" if filename else "违规检测"
    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    safe = filename.replace("/", "_").replace("\\", "_") or "sample"
    fig.savefig(os.path.join(save_dir, f"{safe}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ========================= 入口 =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOP-Vision 对比实验")
    parser.add_argument("--skip_inference", action="store_true",
                        help="跳过 ST-GCN 推理，使用缓存的推理结果")
    args = parser.parse_args()
    run_experiment(skip_inference=args.skip_inference)
