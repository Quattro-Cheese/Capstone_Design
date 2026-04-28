from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class PoseResult:
    image_landmarks: Optional[list]
    world_landmarks: Optional[list]
    timestamp_ms: int


class PoseDetector:
    def __init__(self, model_path: str) -> None:
        self._mp = mp

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr) -> PoseResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms = int(time.time() * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        image_landmarks = result.pose_landmarks if result.pose_landmarks else None
        world_landmarks = (
            result.pose_world_landmarks if result.pose_world_landmarks else None
        )

        return PoseResult(
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self._landmarker.close()
