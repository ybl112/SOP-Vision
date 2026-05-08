"""
MediaPipe Pose 骨架提取封装。

依赖: mediapipe, opencv-python
"""

import numpy as np
from typing import List, Optional


class PoseEstimator:
    """
    MediaPipe Pose 封装，从视频帧提取 33 关键点 3D 坐标。

    Usage:
        estimator = PoseEstimator()
        skeletons = estimator.extract_from_video("video.mp4")  # list of (33, 3)
    """

    def __init__(self, static_image_mode=False, model_complexity=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose = None

    def _get_pose(self):
        if self._pose is None:
            try:
                import mediapipe as mp
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=self.static_image_mode,
                    model_complexity=self.model_complexity,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
            except ImportError:
                raise ImportError("请安装 mediapipe: pip install mediapipe")
        return self._pose

    def extract_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从单帧 BGR 图像提取 33 关键点。

        Returns:
            (33, 3) numpy array 或 None（未检测到人）
        """
        pose = self._get_pose()
        rgb = frame[..., ::-1] if frame.shape[-1] == 3 else frame
        results = pose.process(rgb)

        if results.pose_landmarks is None:
            return None

        landmarks = results.pose_landmarks.landmark
        skeleton = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        return skeleton

    def extract_from_video(self, video_path: str,
                           skip_frames: int = 0) -> List[np.ndarray]:
        """
        从视频文件提取所有帧的骨架序列。

        Args:
            video_path: 视频文件路径
            skip_frames: 跳帧间隔（1=每帧, 2=每2帧）

        Returns:
            [(33, 3), ...] 逐帧骨架列表
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("请安装 opencv-python: pip install opencv-python")

        cap = cv2.VideoCapture(video_path)
        skeletons = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            skel = self.extract_from_frame(frame)
            if skel is not None:
                skeletons.append(skel)
            frame_idx += 1

        cap.release()
        return skeletons

    def save_skeletons(self, skeletons: List[np.ndarray], output_path: str):
        """将骨架序列保存为 .npy 文件 (T, 33, 3)。"""
        data = np.stack(skeletons, axis=0) if skeletons else np.zeros((0, 33, 3))
        np.save(output_path, data)

    def close(self):
        if self._pose is not None:
            self._pose.close()
            self._pose = None
