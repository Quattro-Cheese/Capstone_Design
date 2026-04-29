from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

# MediaPipe Pose Landmark 인덱스 (https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16

# 팔이 곧게 펴진 것으로 판정하는 최소 각도 (AHA 2020 기준 170°, 2D 카메라 오차 감안해 160°로 설정)
ELBOW_ANGLE_THRESHOLD = 160.0

# 압박 중(DOWN) 전환 기준: 이 각도 미만이면 팔이 충분히 굽혀진 것으로 판단
DOWN_THRESHOLD = 130.0

# 복귀(UP) 전환 기준: DOWN 이후 이 각도 이상이면 업스트로크 정점으로 판단 → 자세 판정 시점
UP_THRESHOLD = 150.0

# landmark visibility가 이 값 미만이면 신뢰도가 낮아 각도 계산에서 제외
VISIBILITY_THRESHOLD = 0.5


class CprStage(Enum):
    IDLE = auto()  # 초기 상태 또는 CPR 동작 미감지
    DOWN = auto()  # 압박 중 (팔 굽힘)
    UP = auto()  # 압박 후 복귀 (팔 펴짐) — 이 전환 시점에서 자세를 판정


@dataclass
class PoseEvalResult:
    left_elbow_angle: float  # 왼팔 팔꿈치 각도 (visibility 미달 시 -1.0)
    right_elbow_angle: float  # 오른팔 팔꿈치 각도 (visibility 미달 시 -1.0)
    is_correct: bool  # ELBOW_ANGLE_THRESHOLD 충족 여부
    feedback: str  # 사용자에게 보여줄 피드백 메시지
    stage: CprStage  # 판정 시점의 Stage


class CprStageDetector:
    """매 프레임 팔꿈치 각도를 받아 CPR Stage를 추적하고, DOWN→UP 전환 시점에 자세를 판정한다."""

    def __init__(self) -> None:
        self._stage = CprStage.IDLE
        self._last_result: PoseEvalResult | None = None

    @property
    def stage(self) -> CprStage:
        return self._stage

    @property
    def last_result(self) -> PoseEvalResult | None:
        """가장 최근 판정 결과 — UP 전환 후 다음 판정 전까지 유지됨"""
        return self._last_result

    def update(
        self,
        angle: float,
        left_angle: float | None,
        right_angle: float | None,
    ) -> PoseEvalResult | None:
        """
        Stage를 갱신하고, DOWN→UP 전환 시점에만 PoseEvalResult를 반환한다.
        그 외 프레임에서는 None을 반환한다.
        """
        prev_stage = self._stage

        if angle < DOWN_THRESHOLD:
            self._stage = CprStage.DOWN
        elif angle >= UP_THRESHOLD and prev_stage in (CprStage.DOWN, CprStage.IDLE):
            self._stage = CprStage.UP

        # DOWN→UP 전환 = 업스트로크 정점 → 이 시점에서만 자세 판정
        if self._stage == CprStage.UP and prev_stage == CprStage.DOWN:
            is_correct = angle >= ELBOW_ANGLE_THRESHOLD
            feedback = (
                "자세 정확: 팔이 곧게 펴져 있습니다"
                if is_correct
                else f"팔을 더 펴세요 ({ELBOW_ANGLE_THRESHOLD - angle:.1f}° 부족)"
            )
            self._last_result = PoseEvalResult(
                left_elbow_angle=left_angle if left_angle is not None else -1.0,
                right_elbow_angle=right_angle if right_angle is not None else -1.0,
                is_correct=is_correct,
                feedback=feedback,
                stage=self._stage,
            )
            return self._last_result

        return None

    def reset(self) -> None:
        """landmark를 잃었을 때 Stage를 초기화한다."""
        self._stage = CprStage.IDLE


def _to_pixel(landmark, width: int, height: int) -> np.ndarray:
    """정규화된 landmark 좌표(0~1)를 픽셀 좌표로 변환한다."""
    return np.array([landmark.x * width, landmark.y * height])


def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """세 픽셀 좌표로 꼭짓점 b에서의 각도(0~180°)를 계산한다."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def evaluate_pose(
    image_landmarks,
    frame_width: int,
    frame_height: int,
    visibilities: list[list[float]] | None = None,
    stage_detector: CprStageDetector | None = None,
) -> PoseEvalResult | None:
    """
    팔꿈치 각도를 계산하고 stage_detector에 전달한다.
    stage_detector가 None이면 매 프레임 즉시 판정한다 (테스트용).
    """
    if not image_landmarks:
        if stage_detector is not None:
            stage_detector.reset()
        return None

    lm = image_landmarks[0]

    # 팔 단위로 visibility를 체크해 보이는 팔만 선택
    # 측면 촬영 시 반대쪽 팔은 항상 가려지므로 양팔을 묶어서 체크하면 판정 자체가 불가능해짐
    left_visible = True
    right_visible = True

    if visibilities is not None and len(visibilities) > 0:
        vis = visibilities[0]
        left_visible = all(
            vis[i] >= VISIBILITY_THRESHOLD
            for i in [_LEFT_SHOULDER, _LEFT_ELBOW, _LEFT_WRIST]
        )
        right_visible = all(
            vis[i] >= VISIBILITY_THRESHOLD
            for i in [_RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST]
        )

    if not left_visible and not right_visible:
        if stage_detector is not None:
            stage_detector.reset()
        return None

    left_angle: float | None = None
    right_angle: float | None = None

    if left_visible:
        left_angle = _calculate_angle(
            _to_pixel(lm[_LEFT_SHOULDER], frame_width, frame_height),
            _to_pixel(lm[_LEFT_ELBOW], frame_width, frame_height),
            _to_pixel(lm[_LEFT_WRIST], frame_width, frame_height),
        )
    if right_visible:
        right_angle = _calculate_angle(
            _to_pixel(lm[_RIGHT_SHOULDER], frame_width, frame_height),
            _to_pixel(lm[_RIGHT_ELBOW], frame_width, frame_height),
            _to_pixel(lm[_RIGHT_WRIST], frame_width, frame_height),
        )

    # 보이는 팔 중 더 구부러진 쪽을 기준으로 Stage 판정
    visible_angles = [a for a in [left_angle, right_angle] if a is not None]
    min_angle = min(visible_angles)

    if stage_detector is not None:
        return stage_detector.update(min_angle, left_angle, right_angle)

    # 테스트용 즉시 판정
    is_correct = min_angle >= ELBOW_ANGLE_THRESHOLD
    return PoseEvalResult(
        left_elbow_angle=left_angle if left_angle is not None else -1.0,
        right_elbow_angle=right_angle if right_angle is not None else -1.0,
        is_correct=is_correct,
        feedback=(
            "자세 정확: 팔이 곧게 펴져 있습니다"
            if is_correct
            else f"팔을 더 펴세요 ({ELBOW_ANGLE_THRESHOLD - min_angle:.1f}° 부족)"
        ),
        stage=CprStage.IDLE,
    )
