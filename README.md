# Sign-us Recognition

영상에서 손·포즈·얼굴 랜드마크를 추출하고, 시계열 수어 동작을 분류하기 위해 구성한 학습·평가·실시간 추론 파이프라인입니다.

이 저장소는 포트폴리오용 **source-only 공개본**입니다. 학습에 사용한 원본 영상과 좌표 데이터, 모델 가중치, 개인 환경설정 및 API 키는 포함하지 않습니다.

## What I worked on

- MediaPipe 기반 손·포즈·얼굴 랜드마크 추출
- 좌표 정규화, 관절 각도 및 상대 위치를 결합한 시계열 특징 구성
- Conv1D와 Bidirectional LSTM 기반 분류 모델 학습
- 사람 단위 교차 검증과 holdout 평가
- 노이즈·속도 변화·좌우 반전·프레임 마스킹 기반 데이터 증강
- 앙상블 및 test-time augmentation을 활용한 오류 분석
- 웹캠 입력을 이용한 한국 수어 실시간 추론 실험

## Repository structure

```text
src/
├── asl/
│   ├── extract_keypoints.py  # 영상에서 랜드마크 시퀀스 추출
│   ├── train_final.py        # 최종 10배 증강·사람 단위 4-fold 학습
│   ├── evaluate_ensemble.py  # 최종 4개 fold 앙상블 평가
│   └── experiments/          # 최종 결정 전 260차원 후속 실험 기록
└── ksl/
    ├── train.py              # 64차원 hybrid feature 기반 학습·튜닝
    └── realtime_demo.py      # 웹캠 기반 실시간 추론
```

## Final ASL experiment

실제 최종 모델 생성에 사용한 `ASL_Citizen/asl_final`의 학습·평가 코드를 기준으로 공개본을 구성했습니다.

- 사람 단위 4-fold 교차 검증
- 원본 포함 10배 데이터 증강
- Conv1D + Bidirectional LSTM
- 평균 검증 정확도: **74.58%**
- 단일 best fold holdout 정확도: **74.40%**
- 4-fold ensemble holdout 정확도: **82.86%**

위 수치는 로컬 최종 실험 로그에서 옮긴 값이며, 원본 데이터와 가중치는 재배포하지 않습니다.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

경로는 코드에 개인정보가 남지 않도록 환경변수 또는 저장소 상대경로로 구성했습니다. 필요한 값은 `.env.example`을 참고하세요.

## Data and model policy

- `data/`와 `artifacts/`는 Git에서 제외됩니다.
- 원본 영상, 추출 좌표, 학습된 가중치는 재배포하지 않습니다.
- 데이터셋을 재현에 사용할 경우 각 원본 제공처의 이용 조건과 라이선스를 먼저 확인해야 합니다.
- 모델 가중치가 없어도 코드 구조를 통해 전처리, 특징 설계, 학습, 검증 및 추론 과정을 확인할 수 있습니다.

## Current limitation

공개본은 개인정보와 재배포 제한 자료를 제거했기 때문에 즉시 학습 가능한 완성 데이터셋을 제공하지 않습니다. 로컬에서 적법하게 준비한 데이터와 모델 경로를 지정해야 전체 파이프라인을 실행할 수 있습니다.

## Tech stack

Python · TensorFlow/Keras · MediaPipe · OpenCV · NumPy · scikit-learn · Optuna
