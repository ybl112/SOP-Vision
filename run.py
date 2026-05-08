"""
SOP-Vision 端到端推理实验 — 直接修改下方参数即可运行。

用法:
    python run.py                    # 完整推理 + 生成图表
    python run.py --skip_inference   # 跳过推理，用缓存结果
"""

import sys
import importlib.machinery
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ============================================================
#  推理参数（直接在这里改）
# ============================================================
SKIP_INFERENCE = False    # True=使用缓存, False=完整推理
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SOP-Vision 推理实验")
    parser.add_argument("--skip_inference", action="store_true", default=SKIP_INFERENCE)
    args = parser.parse_args()

    script = str(ROOT / "scripts" / "3_run_experiment.py")
    loader = importlib.machinery.SourceFileLoader("_run_experiment", script)
    run_mod = loader.load_module()
    run_mod.run_experiment(skip_inference=args.skip_inference)
