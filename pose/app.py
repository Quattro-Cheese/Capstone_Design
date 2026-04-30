from __future__ import annotations

from pathlib import Path
import cv2

from pose.detector import PoseDetector
from pose.evaluator import HysteresisJudge, evaluate_pose
from pose.visualizer import draw_eval_result, draw_pose_points

MAX_FRAME_FAILURES = 10


def main() -> None:
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "models" / "pose_landmarker_full.task"

    if not model_path.exists():
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return

    detector = PoseDetector(model_path=str(model_path))
    hysteresis = HysteresisJudge()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    consecutive_failures = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FRAME_FAILURES:
                    print("카메라 프레임을 읽을 수 없습니다.")
                    break
                continue
            consecutive_failures = 0

            frame = cv2.flip(frame, 1)

            pose_result = detector.detect(frame)
            draw_pose_points(frame, pose_result.image_landmarks)

            eval_result = evaluate_pose(
                pose_result.image_landmarks,
                pose_result.frame_width,
                pose_result.frame_height,
                pose_result.visibilities,
                hysteresis,
            )

            if eval_result is not None:
                draw_eval_result(frame, eval_result)

            cv2.putText(
                frame,
                "Tip: Position camera so arm is fully visible from the side",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("CPR Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
