from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe Pose Landmark 인덱스
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16

# CPR 자세 기준 (AHA 2020 가이드라인: 팔을 곧게 펴서 체중으로 압박)
ELBOW_ANGLE_THRESHOLD = 170.0  # 도(degree)


@dataclass
class PoseEvalResult:
    left_elbow_angle: float  # 왼쪽 팔꿈치 각도
    right_elbow_angle: float  # 오른쪽 팔꿈치 각도
    is_correct: bool  # 기준 충족 여부
    feedback: str  # 피드백 메시지


def _to_array(landmark) -> np.ndarray:
    """MediaPipe world landmark → numpy array (x, y, z)"""
    return np.array([landmark.x, landmark.y, landmark.z])


def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    세 점의 각도 계산 (꼭짓점: b)

    world_landmarks를 사용하는 이유:
    - image_landmarks: 2D 픽셀 좌표 → 카메라 각도에 따라 각도 왜곡 발생
    - world_landmarks: 3D 실세계 좌표(미터 단위) → 카메라 위치 무관하게 실제 관절 각도 측정 가능

    Args:
        a: 시작점 (어깨)
        b: 꼭짓점 (팔꿈치)
        c: 끝점 (손목)

    Returns:
        0~180 사이의 각도 (degree)
    """
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return float(angle)


def evaluate_pose(world_landmarks) -> PoseEvalResult | None:
    """
    world_landmarks로부터 팔꿈치 각도를 계산하고 자세를 평가한다.

    판정 기준:
    - 양쪽 팔꿈치 각도를 모두 측정
    - 더 낮은 쪽(더 구부러진 쪽)을 기준으로 판정 (보수적 평가)
    - ELBOW_ANGLE_THRESHOLD(170°) 이상이면 올바른 자세로 판정

    Args:
        world_landmarks: PoseDetector.detect()의 PoseResult.world_landmarks
                         (None이면 None 반환)

    Returns:
        PoseEvalResult 또는 None (landmark 미검출 시)
    """
    if not world_landmarks:
        return None

    lm = world_landmarks[0]  # 첫 번째 사람

    left_angle = _calculate_angle(
        _to_array(lm[_LEFT_SHOULDER]),
        _to_array(lm[_LEFT_ELBOW]),
        _to_array(lm[_LEFT_WRIST]),
    )
    right_angle = _calculate_angle(
        _to_array(lm[_RIGHT_SHOULDER]),
        _to_array(lm[_RIGHT_ELBOW]),
        _to_array(lm[_RIGHT_WRIST]),
    )

    # 더 구부러진 팔 기준으로 판정
    min_angle = min(left_angle, right_angle)
    is_correct = min_angle >= ELBOW_ANGLE_THRESHOLD

    if is_correct:
        feedback = "자세 정확: 팔이 곧게 펴져 있습니다"
    else:
        diff = ELBOW_ANGLE_THRESHOLD - min_angle
        feedback = f"팔을 더 펴세요 ({diff:.1f}° 부족)"

    return PoseEvalResult(
        left_elbow_angle=left_angle,
        right_elbow_angle=right_angle,
        is_correct=is_correct,
        feedback=feedback,
    )
