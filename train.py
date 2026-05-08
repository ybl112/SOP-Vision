"""
SOP-Vision 训练脚本 — 直接修改下方参数即可运行。

用法:
    python train.py                    # 使用下方参数
    python train.py --epochs 100       # 命令行可临时覆盖
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config_loader import load_config
from src.perception.train import train_model

# ============================================================
#  训练参数（直接在这里改）
# ============================================================
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 0.001
WEIGHT_DECAY = 0.0001
WINDOW_SIZE  = 32
# ============================================================

if __name__ == "__main__":
    import argparse

    cfg = load_config()
    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    parser = argparse.ArgumentParser(description="SOP-Vision 训练")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--data_dir", type=str,
                        default=ds_cfg.get("skeleton_dir", "data/skeletons/"))
    parser.add_argument("--train_csv", type=str,
                        default=ds_cfg.get("label_csv_dir", "data/labels/") + "train.csv")
    parser.add_argument("--test_csv", type=str, default=None)
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_classes=ds_cfg.get("num_classes", 7),
        in_channels=model_cfg.get("in_channels", 3),
        save_path=train_cfg.get("save_dir", "outputs/models/") + "best_model.pt",
    )
