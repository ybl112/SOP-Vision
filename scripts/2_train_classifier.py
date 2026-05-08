"""
步骤2：训练 ST-GCN 工序分类器。

用法:
    python scripts/2_train_classifier.py --data_dir data/skeletons --train_csv data/labels/train.csv
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src.perception.train import train_model


def main():
    cfg = load_config()
    ds_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    parser = argparse.ArgumentParser(description="Training ST-GCN classifier")
    parser.add_argument("--data_dir", type=str, default=ds_cfg.get("skeleton_dir", "data/skeletons/"))
    parser.add_argument("--train_csv", type=str,
                        default=os.path.join(ds_cfg.get("label_csv_dir", "data/labels/"), "train.csv"))
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=model_cfg.get("window_size", 32))
    parser.add_argument("--batch_size", type=int, default=train_cfg.get("batch_size", 32))
    parser.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 50))
    parser.add_argument("--lr", type=float, default=train_cfg.get("lr", 0.001))
    parser.add_argument("--weight_decay", type=float, default=train_cfg.get("weight_decay", 0.0001))
    parser.add_argument("--save_path", type=str,
                        default=train_cfg.get("save_dir", "outputs/models/") + "best_model.pt")
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
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
