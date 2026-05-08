"""
步骤1：从原始视频中提取 MediaPipe 33 关键点骨架序列。

用法:
    python scripts/1_extract_skeletons.py --video_dir data/raw_videos --out_dir data/skeletons

输出:
    每个视频生成一个 .npy 文件，shape = (T, 99)，即 T 帧 × 99 个坐标值(33点 × 3维x,y,z)。
"""

import os
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

from src.config_loader import load_config

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

POSE = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,        # 0=Lite, 1=Full, 2=Heavy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_skeleton(video_path, out_path, show_preview=False):
    """
    从视频文件中逐帧提取 33 关键点坐标。

    Args:
        video_path:   输入视频路径
        out_path:     输出 .npy 路径
        show_preview: 是否显示带骨架的可视化窗口

    Returns:
        frame_count: 有效帧数
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] 无法打开视频文件: {video_path}")
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

            if show_preview:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow("Skeleton Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    if not skeletons:
        print(f"  [WARN] 未提取到任何骨架: {video_path}")
        return 0

    arr = np.array(skeletons, dtype=np.float32)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, arr)
    return len(skeletons)


def main():
    cfg = load_config()
    ds_cfg = cfg.get("dataset", {})

    parser = argparse.ArgumentParser(description="MediaPipe 骨架提取")
    parser.add_argument("--video_dir", type=str,
                        default=ds_cfg.get("raw_video_dir", "data/raw_videos/"),
                        help="原始视频目录")
    parser.add_argument("--out_dir", type=str,
                        default=ds_cfg.get("skeleton_dir", "data/skeletons/"),
                        help="骨架 .npy 输出目录")
    parser.add_argument("--ext", type=str, default=".mp4",
                        help="视频文件扩展名，默认 .mp4")
    parser.add_argument("--show", action="store_true",
                        help="显示骨架可视化预览")
    args = parser.parse_args()

    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"视频目录不存在: {args.video_dir}")

    video_files = sorted([
        f for f in os.listdir(args.video_dir)
        if f.lower().endswith(args.ext)
    ])

    if not video_files:
        print(f"未在 {args.video_dir} 中找到 {args.ext} 文件。")
        return

    print(f"找到 {len(video_files)} 个视频，开始提取骨架 ...")
    total_frames = 0

    for vf in tqdm(video_files, desc="提取骨架"):
        video_path = os.path.join(args.video_dir, vf)
        out_path = os.path.join(args.out_dir, Path(vf).stem + ".npy")
        n = extract_skeleton(video_path, out_path, show_preview=args.show)
        total_frames += n

    print(f"完成。共 {total_frames} 帧保存至 {args.out_dir}")


if __name__ == "__main__":
    main()
