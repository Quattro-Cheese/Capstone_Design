from __future__ import annotations

import cv2

from pose.evaluator import PoseEvalResult


def draw_pose_points(frame, image_landmarks) -> None:
    if not image_landmarks:
        return

    h, w, _ = frame.shape
    first_pose = image_landmarks[0]

    for lm in first_pose:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def draw_eval_result(frame, result: PoseEvalResult) -> None:
    """
    팔꿈치 각도 및 자세 판정 결과를 화면 좌상단에 오버레이한다.

    표시 내용:
    - 왼쪽/오른쪽 팔꿈치 각도
    - 자세 정확 여부 (초록/빨강)
    - 피드백 메시지
    """
    COLOR_OK = (0, 255, 0)  # 초록: 기준 충족
    COLOR_BAD = (0, 0, 255)  # 빨강: 기준 미달
    COLOR_INFO = (255, 255, 255)  # 흰색: 정보

    color = COLOR_OK if result.is_correct else COLOR_BAD

    lines = [
        (f"L elbow: {result.left_elbow_angle:.1f}deg", COLOR_INFO),
        (f"R elbow: {result.right_elbow_angle:.1f}deg", COLOR_INFO),
        (result.feedback, color),
    ]

    x, y_start, dy = 10, 30, 30
    for text, c in lines:
        cv2.putText(frame, text, (x, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        y_start += dy
