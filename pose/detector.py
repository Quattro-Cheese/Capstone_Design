from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class PoseResult:
    image_landmarks: Optional[list]
    visibilities: list[list[float]] = field(
        default_factory=list
    )  # [사람 인덱스][landmark 인덱스] = visibility(0~1)
    frame_width: int = 0
    frame_height: int = 0
    timestamp_ms: int = 0


class PoseDetector:
    def __init__(self, model_path: str) -> None:
        self._mp = mp
        self._start_time = time.monotonic()  # timestamp 증가를 위해 시작 시각 기록

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
        frame_height, frame_width = frame_bgr.shape[:2]

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        # MediaPipe VIDEO 모드는 timestamp가 항상 이전 값보다 커야 하므로 monotonic 사용
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        image_landmarks = result.pose_landmarks if result.pose_landmarks else None

        # visibility는 image_landmarks에만 있으므로 별도로 추출해 evaluator에 전달
        visibilities: list[list[float]] = []
        if image_landmarks:
            for pose in image_landmarks:
                visibilities.append([lm.visibility for lm in pose])

        return PoseResult(
            image_landmarks=image_landmarks,
            visibilities=visibilities,
            frame_width=frame_width,
            frame_height=frame_height,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self._landmarker.close()
