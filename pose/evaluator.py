from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16

ELBOW_ANGLE_THRESHOLD = 170.0  # degree

# MediaPipe visibility 신뢰도 기준
# 0~1 사이 값, 0.5 미만이면 가려지거나 추정 신뢰도가 낮은 landmark
VISIBILITY_THRESHOLD = 0.5

# 각도 계산에 필요한 landmark 인덱스 (양팔 어깨·팔꿈치·손목)
_ARM_LANDMARK_INDICES = [
    _LEFT_SHOULDER,
    _LEFT_ELBOW,
    _LEFT_WRIST,
    _RIGHT_SHOULDER,
    _RIGHT_ELBOW,
    _RIGHT_WRIST,
]


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


def evaluate_pose(
    world_landmarks,
    visibilities: list[list[float]] | None = None,
) -> PoseEvalResult | None:
    """
    world_landmarks로부터 팔꿈치 각도를 계산하고 자세를 평가한다.

    각도 계산과 visibility 판단을 분리하는 이유:
    - 각도 계산: world_landmarks 사용 (3D 실세계 좌표, 카메라 각도 왜곡 없음)
    - visibility 판단: image_landmarks의 visibility 사용 (world_landmarks에는 없음)

    판정 기준:
    - 6개 landmark(양팔 어깨·팔꿈치·손목) visibility 모두 0.5 이상이어야 판정
    - 양쪽 팔꿈치 각도를 모두 측정
    - 더 낮은 쪽(더 구부러진 쪽)을 기준으로 판정 (보수적 평가)
    - ELBOW_ANGLE_THRESHOLD(170°) 이상이면 올바른 자세로 판정

    Args:
        world_landmarks: PoseDetector.detect()의 PoseResult.world_landmarks
        visibilities: PoseDetector.detect()의 PoseResult.visibilities
                      (None이면 visibility 체크 생략)

    Returns:
        PoseEvalResult 또는 None (landmark 미검출 또는 visibility 미달 시)
    """
    if not world_landmarks:
        return None

    lm = world_landmarks[0]  # 첫 번째 사람

    # visibility 체크: image_landmarks의 visibility로 신뢰도 판단
    # visibilities가 전달된 경우에만 실제 체크 수행
    if visibilities:
        vis = visibilities[0]  # 첫 번째 사람의 visibility 리스트
        if not all(vis[i] >= VISIBILITY_THRESHOLD for i in _ARM_LANDMARK_INDICES):
            return None

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
