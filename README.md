# CPR MediaPipe Pose Module

## 0. Overview

### MediaPipe
- 실시간 영상에서 landmark(관절 좌표)를 추출하는 라이브러리

### Pose Landmark
- 인체를 33개의 keypoint로 표현
- 각 landmark는 (x, y, z) 좌표를 가짐
- CPR 자세 분석의 입력 데이터로 사용

---

## 1. Setup

### 1.1 프로젝트 이동
```bash
cd cpr_cheese
```

### 1.2 가상환경 생성 
```bash
python -m venv venv
# 초기 세팅 시 한번만 하면 OK
```

### 1.3 가상환경 활성화
```bash
# Git Bash
source venv/Scripts/activate

# Windows (CMD / PowerShell)
venv/Scripts/activate

# 매번 실행해야하는 부분 
# [(venv) anary@□□□□□□ MINGW6] 가 나와야 정상

```

### 1.4 패키지 설치
```bash
pip install -r requirements.txt
```

- mediapipe / opencv-python / numpy 설치
- 라이브러리 추가 시 requirements.txt 수정

### 1.5 실행
```bash
python app.py
```

---

## 2. Model

```text
models/pose_landmarker.task
```

- MediaPipe에서 사용하는 사전 학습 모델 파일
- 반드시 필요 (없으면 실행 불가)

### 모델 종류
- pose_landmarker_lite  
  → 빠름 / 실시간 최적화 / 정확도 낮음

- pose_landmarker_full  
  → 속도-정확도 균형

- pose_landmarker_heavy  
  → 정확도 높음 / 매우 느림 (비추천)

### 모델 변경 방법
```python
# app.py
model_path = "models/pose_landmarker_lite.task"
```

---

## 3. Project Structure

```text
cpr_cheese/
├── app.py
├── models/
│   └── pose_landmarker.task
├── pose/
│   ├── detector.py        # MediaPipe 실행 및 landmark 추출
│   └── visualizer.py      # landmark 시각화
└── requirements.txt
```

---

## 4. Output

입력:
- 웹캠 영상

출력:
- pose_landmarks (2D)
- pose_world_landmarks (3D)

---

## 5. Scope

현재 구현:
```text
MediaPipe 연동
Pose Landmark 추출
실시간 시각화
```

미구현:
```text
각도 계산
CPR 자세 평가
BPM 분석
피드백 시스템
```

---

## 6. Note

- 본 모듈은 "Pose Estimation 결과 생성"까지 담당
- 이후 로직은 landmark 데이터를 기반으로 구현
