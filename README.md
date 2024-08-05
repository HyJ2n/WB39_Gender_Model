Relearning a model that can be gender-division using the Yolov8x model.
Multiple gender face datasets are required.

label_Reclass.py : Modify the gender database.txt file consisting of class numbers and coordinate values to the desired class number.

Yolo 모델 기반 : gender_model2.py
Test : test_gende2.py

ResNet 모델 기반 : gender_model.py
Test : test_gender.py
=> 기존 Age모델과 동일한 캐싱 파일 사용 , Yolo모델보다 높은 정확성 제공
