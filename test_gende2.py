from ultralytics import YOLO
from PIL import Image

# 모델 로드
model = YOLO('gender_model.pt')


# 예측할 이미지 경로 설정
image_source = r"C:\Users\admin\Desktop\Project\2we\yolov8\17.png"
image = Image.open(image_source)

# 이미지 예측
results = model.predict(
    source=image,
    save=True,
    conf=0.6
    )

# 결과 분석
# 예측 결과에서 원하는 정보를 추출합니다.
# 각 감지된 객체의 클래스 인덱스를 확인합니다.
if results and len(results) > 0 and len(results[0].boxes) > 0:
    for box in results[0].boxes:
        label = int(box.cls)  # 예측된 클래스 인덱스를 정수로 변환하여 가져옵니다.
        
        if label == 0:
            print("Gender: Male")
        elif label == 1:
            print("Gender: Female")
        else:
            print("Unknown label:", label)
else:
    print("No predictions were made.")