from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe Pose Landmark 인덱스 (https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16

# 히스테리시스 임계값
# 논문 근거: Weiss et al. (2023), Advances in Simulation — CPR 포즈 추정 연구에서 170°를 기준으로 사용
# 단, MediaPipe 2D 단일 카메라의 측정 오차(±8°)를 감안해 상하한을 분리 설정
# 실측 데이터 기반으로 마네킹 대여 기간(4/14~4/20)에 재조정 예정
CORRECT_THRESHOLD = 163.0  # 이 각도 이상이면 "정확"으로 전환
INCORRECT_THRESHOLD = 157.0  # 이 각도 미만이면 "부정확"으로 전환
# 157°~163° 구간에서는 이전 상태를 유지해 측정 노이즈로 인한 판정 깜빡임을 방지

# landmark visibility가 이 값 미만이면 신뢰도가 낮아 각도 계산에서 제외
VISIBILITY_THRESHOLD = 0.5


@dataclass
class PoseEvalResult:
    left_elbow_angle: float  # 왼팔 팔꿈치 각도 (visibility 미달 시 -1.0)
    right_elbow_angle: float  # 오른팔 팔꿈치 각도 (visibility 미달 시 -1.0)
    is_correct: bool  # 팔꿈치 각도가 기준을 충족하는지 여부
    feedback: str  # 사용자에게 보여줄 피드백 메시지


class HysteresisJudge:
    """
    히스테리시스 방식으로 팔꿈치 각도를 판정한다.

    히스테리시스란:
      "정확" 판정으로 바뀌는 기준(CORRECT_THRESHOLD)과
      "부정확" 판정으로 바뀌는 기준(INCORRECT_THRESHOLD)을 다르게 설정하는 방식.

      예시 (CORRECT=163°, INCORRECT=157°):
        현재 부정확 상태 → 163° 이상이 돼야 정확으로 전환
        현재 정확 상태   → 157° 미만이 돼야 부정확으로 전환
        157°~163° 구간   → 이전 상태 유지 (전환 없음)

      이렇게 하면 MediaPipe 측정 노이즈(±5° 내외)로 인해
      판정이 매 프레임 초록/빨강으로 깜빡이는 문제를 방지할 수 있다.
    """

    def __init__(self) -> None:
        # 초기 상태는 부정확으로 시작 (측정 전 상태)
        self._is_correct = False

    def update(self, angle: float) -> bool:
        """
        현재 각도로 판정 상태를 갱신하고 결과를 반환한다.

        - 현재 부정확 상태: angle >= CORRECT_THRESHOLD 이면 정확으로 전환
        - 현재 정확 상태:   angle <  INCORRECT_THRESHOLD 이면 부정확으로 전환
        - 그 사이 구간:     이전 상태 유지
        """
        if not self._is_correct and angle >= CORRECT_THRESHOLD:
            self._is_correct = True
        elif self._is_correct and angle < INCORRECT_THRESHOLD:
            self._is_correct = False
        return self._is_correct

    def reset(self) -> None:
        """landmark를 잃었을 때 판정 상태를 초기화한다."""
        self._is_correct = False


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
    hysteresis: HysteresisJudge | None = None,
) -> PoseEvalResult | None:
    """
    팔꿈치 각도를 계산하고 히스테리시스 판정기로 자세를 평가한다.
    hysteresis가 None이면 단순 임계값으로 즉시 판정한다 (테스트용).
    """
    if not image_landmarks:
        if hysteresis is not None:
            hysteresis.reset()
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
        if hysteresis is not None:
            hysteresis.reset()
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

    # 보이는 팔 중 더 구부러진 쪽을 기준으로 판정
    visible_angles = [a for a in [left_angle, right_angle] if a is not None]
    min_angle = min(visible_angles)

    # 히스테리시스 판정
    if hysteresis is not None:
        is_correct = hysteresis.update(min_angle)
    else:
        # 테스트용 단순 임계값 판정
        is_correct = min_angle >= CORRECT_THRESHOLD

    feedback = (
        "Good: Arms are straight"
        if is_correct
        else f"Straighten your arms ({CORRECT_THRESHOLD - min_angle:.1f}deg short)"
    )

    return PoseEvalResult(
        left_elbow_angle=left_angle if left_angle is not None else -1.0,
        right_elbow_angle=right_angle if right_angle is not None else -1.0,
        is_correct=is_correct,
        feedback=feedback,
    )
