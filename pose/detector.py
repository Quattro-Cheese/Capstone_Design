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
    world_landmarks: Optional[list]
    # image_landmarks에서 추출한 landmark별 visibility 리스트
    # 각도 계산은 world_landmarks로, 신뢰도 판단은 image_landmarks로 수행하기 위해 분리
    # shape: List[List[float]] — [사람 인덱스][landmark 인덱스] = visibility(0~1)
    visibilities: list[list[float]] = field(default_factory=list)
    timestamp_ms: int = 0


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

        # image_landmarks에서 visibility 추출
        # world_landmarks에는 visibility가 없으므로 image_landmarks의 visibility를 별도로 넘겨 evaluator에서 신뢰도 판단에 사용
        visibilities: list[list[float]] = []
        if image_landmarks:
            for pose in image_landmarks:
                visibilities.append([lm.visibility for lm in pose])

        return PoseResult(
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
            visibilities=visibilities,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self._landmarker.close()
