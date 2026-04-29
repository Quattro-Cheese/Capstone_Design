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
# 0~1 사이 값, 0.5 미만이면 가려지거나 추정 신뢰도가 낮은 landmark로 간주
VISIBILITY_THRESHOLD = 0.5


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

    if not world_landmarks:
        return None

    lm = world_landmarks[0]  # 첫 번째 사람

    # 팔 단위 visibility 체크
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
        return None

    # visibility 통과한 팔만 각도 계산
    left_angle: float | None = None
    right_angle: float | None = None

    if left_visible:
        left_angle = _calculate_angle(
            _to_array(lm[_LEFT_SHOULDER]),
            _to_array(lm[_LEFT_ELBOW]),
            _to_array(lm[_LEFT_WRIST]),
        )
    if right_visible:
        right_angle = _calculate_angle(
            _to_array(lm[_RIGHT_SHOULDER]),
            _to_array(lm[_RIGHT_ELBOW]),
            _to_array(lm[_RIGHT_WRIST]),
        )

    # visibility 통과한 팔 중 더 구부러진 쪽 기준으로 판정
    visible_angles = [a for a in [left_angle, right_angle] if a is not None]
    min_angle = min(visible_angles)
    is_correct = min_angle >= ELBOW_ANGLE_THRESHOLD

    if is_correct:
        feedback = "자세 정확: 팔이 곧게 펴져 있습니다"
    else:
        diff = ELBOW_ANGLE_THRESHOLD - min_angle
        feedback = f"팔을 더 펴세요 ({diff:.1f}° 부족)"

    return PoseEvalResult(
        left_elbow_angle=left_angle if left_angle is not None else -1.0,
        right_elbow_angle=right_angle if right_angle is not None else -1.0,
        is_correct=is_correct,
        feedback=feedback,
    )
