import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from gender_model import GenderDataset, model1, device, CFG, test_transform
from tqdm import tqdm
import face_recognition as fr
import glob
import os

# 얼굴을 인식하고 잘라내는 함수
def detect_face(image_path):
    image = fr.load_image_file(image_path)
    face_locations = fr.face_locations(image)
    
    if len(face_locations) == 0:
        print(f"얼굴이 인식되지 않습니다: {image_path}")
        return None
    
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    return face_image, (top, right, bottom, left)

# 성별 예측 함수
def predict_gender(model, face_image):
    # 이미지를 Dataset 형식으로 변환
    image = Image.fromarray(face_image)
    face_tensor = test_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logit = model(face_tensor)
        pred = logit.argmax(dim=1, keepdim=True)
        return logit, pred

if __name__ == "__main__":
    person_path = r'C:\Users\admin\Desktop\Gender\test\*.jpg'  # 테스트 이미지 경로
    result_path = r'C:\Users\admin\Desktop\Gender\result'  # 결과 저장 경로

    # 결과 저장 폴더 생성
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    image_paths = glob.glob(person_path)
    
    # 모델 불러오기
    check_point = torch.load(r'C:\Users\admin\Desktop\Gender\gender_best.pth')
    model = model1
    model = model.to(device)
    model.load_state_dict(check_point)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        
        # 얼굴 인식 및 얼굴 이미지 추출
        face_image, face_location = detect_face(img_path)
        if face_image is not None:
            # 성별 예측
            logit, pred = predict_gender(model, face_image)
            pred = pred.tolist()
            logit = logit.tolist()
            
            # 예측 결과 출력
            if pred[0][0] == 0:
                label = "Male"
                confidence = logit[0][pred[0][0]] * 100
            elif pred[0][0] == 1:
                label = "Female"
                confidence = logit[0][pred[0][0]] * 100

            # 예측 결과를 이미지에 표시
            top, right, bottom, left = face_location
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, f'{label}: {confidence:.2f}%', (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 결과 이미지를 result 폴더에 저장
            save_path = os.path.join(result_path, os.path.basename(img_path))
            cv2.imwrite(save_path, img)
            print(f'결과 저장: {save_path}')
