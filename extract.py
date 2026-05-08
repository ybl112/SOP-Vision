"""
SOP-Vision 骨架提取脚本 — 直接修改下方参数即可运行。

用法:
    python extract.py                  # 使用下方参数
    python extract.py --show           # 命令行可临时覆盖
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from src.config_loader import load_config

_cfg = load_config()
_ds = _cfg.get("dataset", {})

# ============================================================
#  提取参数（直接在这里改，默认值来自 config.yaml）
# ============================================================
VIDEO_DIR    = _ds.get("raw_video_dir", "data/raw_videos")
OUT_DIR      = _ds.get("skeleton_dir", "data/skeletons")
VIDEO_EXT    = ".mp4"
SHOW_PREVIEW = False
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

POSE = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_one(video_path, out_path, show=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] 无法打开: {video_path}")
        return 0
    skeletons = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = POSE.process(rgb)
        if results.pose_landmarks:
            coords = []
            for lm in results.pose_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            skeletons.append(coords)
            if show:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow("Skeleton Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    if not skeletons:
        print(f"  [WARN] 未提取到骨架: {video_path}")
        return 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, np.array(skeletons, dtype=np.float32))
    return len(skeletons)


if __name__ == "__main__":
    import argparse

    cfg = load_config()
    ds_cfg = cfg.get("dataset", {})

    parser = argparse.ArgumentParser(description="SOP-Vision 骨架提取")
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--ext", type=str, default=VIDEO_EXT)
    parser.add_argument("--show", action="store_true", default=SHOW_PREVIEW)
    args = parser.parse_args()

    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"视频目录不存在: {args.video_dir}")

    video_files = sorted(f for f in os.listdir(args.video_dir) if f.lower().endswith(args.ext))
    if not video_files:
        print(f"未在 {args.video_dir} 中找到 {args.ext} 文件。")
        sys.exit(0)

    print(f"找到 {len(video_files)} 个视频，开始提取骨架 ...")
    total = 0
    for vf in tqdm(video_files, desc="提取骨架"):
        vp = os.path.join(args.video_dir, vf)
        op = os.path.join(args.out_dir, Path(vf).stem + ".npy")
        total += extract_one(vp, op, show=args.show)
    print(f"完成。共 {total} 帧保存至 {args.out_dir}")
