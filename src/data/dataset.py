"""
PyTorch Dataset / DataLoader for ST-GCN skeleton sequences.

CSV 格式:
  文件名,帧范围标注,工序顺序
  帧范围: label:start-end  用分号分隔
  例: demo_01.npy,0:0-12;1:13-32;2:33-52;3:53-72;4:73-92;5:93-110;6:111-113

骨架 .npy: shape = (T, 99) 即 33 关键点 × 3 坐标(x,y,z)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _parse_frame_ranges(ranges_str: str, expected_frames: int) -> np.ndarray:
    """
    将帧范围字符串转为逐帧标签数组。

    Args:
        ranges_str: "0:0-12;1:13-32;..."
        expected_frames: 骨架总帧数，用于截断/填充

    Returns:
        labels: (T,) int array
    """
    labels = np.zeros(expected_frames, dtype=int)
    for seg in ranges_str.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            label_str, frame_range = seg.split(":")
            label = int(label_str)
            start_str, end_str = frame_range.split("-")
            start, end = int(start_str), int(end_str)
            end = min(end + 1, expected_frames)
            if start < expected_frames:
                labels[start:end] = label
        except (ValueError, IndexError):
            continue
    return labels


class SkeletonDataset(Dataset):

    def __init__(self, data_dir, csv_path, window_size=32, stride=16):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self._samples = []
        self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

        for line in lines:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            filename = parts[0].strip()
            ranges_str = parts[1].strip()

            npy_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(npy_path):
                continue

            data = np.load(npy_path)
            T = data.shape[0]
            labels = _parse_frame_ranges(ranges_str, T)

            for start in range(0, T - self.window_size + 1, self.stride):
                # 取窗口中最多的标签作为该片段标签
                window_labels = labels[start:start + self.window_size]
                majority_label = int(np.bincount(window_labels).argmax())
                self._samples.append((npy_path, start, majority_label))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        npy_path, start, label = self._samples[idx]
        data = np.load(npy_path)
        clip = data[start:start + self.window_size, :]

        T = clip.shape[0]
        clip = clip.reshape(T, 33, 3).transpose(2, 0, 1).astype(np.float32)

        return torch.from_numpy(clip), torch.tensor(label, dtype=torch.long)


def get_dataloader(data_dir, csv_path, batch_size=32, split="train", window_size=32):
    stride = window_size // 2 if split == "train" else window_size
    dataset = SkeletonDataset(data_dir, csv_path, window_size, stride)
    shuffle = (split == "train")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=True)
