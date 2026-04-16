from __future__ import annotations

import cv2


def draw_pose_points(frame, image_landmarks) -> None:
    if not image_landmarks:
        return

    h, w, _ = frame.shape
    first_pose = image_landmarks[0]

    for lm in first_pose:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)