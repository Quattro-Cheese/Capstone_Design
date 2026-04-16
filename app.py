from __future__ import annotations

from pathlib import Path
import cv2

from pose.detector import PoseDetector
from pose.visualizer import draw_pose_points


def main() -> None:
    project_root = Path(__file__).resolve().parent
    #model_path = project_root / "models" / "pose_landmarker_lite.task"
    model_path = project_root / "models" / "pose_landmarker_full.task"
    
    if not model_path.exists():
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return

    detector = PoseDetector(model_path=str(model_path))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)

            pose_result = detector.detect(frame)
            draw_pose_points(frame, pose_result.image_landmarks)

            cv2.imshow("CPR MediaPipe Integration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()