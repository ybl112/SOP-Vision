"""
步骤2：训练 ST-GCN 工序分类器。

用法:
    python scripts/2_train_classifier.py --data_dir data/skeletons --train_csv data/labels/train.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.train import train_model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Training ST-GCN classifier")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_path", type=str, default="outputs/models/best_model.pt")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
