from __future__ import annotations

import cv2

from pose.evaluator import PoseEvalResult

# 화면에 표시할 landmark 인덱스 (어깨·팔꿈치·손목만 표시, 얼굴 등 불필요한 점 제외)
_DRAW_INDICES = {11, 12, 13, 14, 15, 16}

COLOR_OK = (0, 255, 0)  # 초록: 자세 정확
COLOR_BAD = (0, 0, 255)  # 빨강: 자세 미달
COLOR_INFO = (255, 255, 255)  # 흰색: 각도 수치


def draw_pose_points(frame, image_landmarks) -> None:
    """어깨·팔꿈치·손목 landmark를 화면에 점으로 표시한다."""
    if not image_landmarks:
        return

    h, w, _ = frame.shape
    for idx, lm in enumerate(image_landmarks[0]):
        if idx not in _DRAW_INDICES:
            continue
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, (0, 255, 0), -1)


def draw_eval_result(frame, result: PoseEvalResult) -> None:
    """팔꿈치 각도와 피드백 메시지를 좌상단에 표시한다. 각도가 -1.0이면 해당 팔은 표시 생략한다."""
    color = COLOR_OK if result.is_correct else COLOR_BAD

    lines = []
    if result.left_elbow_angle >= 0:
        lines.append((f"L elbow: {result.left_elbow_angle:.1f}deg", COLOR_INFO))
    if result.right_elbow_angle >= 0:
        lines.append((f"R elbow: {result.right_elbow_angle:.1f}deg", COLOR_INFO))
    lines.append((result.feedback, color))

    y = 30
    for text, c in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        y += 30
