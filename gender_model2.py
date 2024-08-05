from ultralytics import YOLO

# 모델 로드 (사전 학습된 모델 사용)
model = YOLO('yolov8x.pt')

# 데이터셋 경로 설정
dataset_yaml = r'C:\Users\admin\Desktop\Project\Main\Gender\data.yaml'

# 모델 학습
results = model.train(
    data=dataset_yaml,
    imgsz=128,
    epochs=50,  # 학습 에폭 수
    batch=5,   # 배치 사이즈
    workers=0,  # 데이터 로드 워커 수
    patience=0 # 조기 종료를 위한 인내 값
)

# 학습된 모델 저장
model.save('gender_model.pt')

# 검증
results = model.val()

# 예측 (새로운 이미지에 대해 예측 수행)
results = model.predict(
    source=r'C:\Users\admin\Desktop\Project\Main\Gender\gender\test',
    save=True
)