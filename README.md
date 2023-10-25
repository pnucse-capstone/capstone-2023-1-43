# 라즈베리 파이를 이용한 스마트폰 부품 판별 시스템

## 1. 프로젝트 소개
- 프로젝트 명 : 라즈베리 파이를 이용한 스마트폰 부품 판별 시스템
- 프로젝트 목적 : 시간이 지남에 따라 추가되는 부품을 더 효율적으로 추가할 수 있는 SNN을 이용한 모델을 선보입니다. 
- 프로젝트 개요 : 기존 CNN 모델로 이러한 객체 판별 모델을 제작 시, 많은 학습 데이터를 요구하기에 매번 추가되는 새 부품을 <br/>더 효율적으로 추가할 수 있는 SNN 모델을 이용하여 판별을 가능케 하였습니다. 
## 2. 팀 소개
### Team Return 0
- 윤태훈, thoon19@naver.com, 데이터 수집 및 전처리, SNN 모델 제작
- 이걍윤, kangyun205@naver.com, 모델 테스트 및 UI 제작

## 3. 구성도
![Untitled](/assets/flow.png)  

1. 투입된 부품을 촬영합니다 / 이미지 선택 이용 시 촬영과 2단계는 스킵합니다.
2. 촬영된 이미지는 2464 * 2464 크기의 RGB 3채널 JPG 이미지로 저장 후, Pillow를 이용하여 96 * 96, 그레이스케일 이미지로 변환합니다.
3. 변환된 입력 이미지 혹은 선택된 이미지를 Tensorflow로 제작된 SNN 모델을 통해 부품 판별 진행합니다.
4. 판정된 결과를 사용자에게 가시적인 형태로 전달합니다.


## 4. 소개 및 시연 영상
### Siamese Neural Network
![SNN](/assets/SNN.png)
SNN은 위 그림에서 설명하는 것과 같이, weight를 공유하는 neural network입니다.<br/>
이 네트워크를 input 이미지가 통과하게 되면, embedding 과정을 거쳐, distance가 산출됩니다.<br/>
이 distance 값을 이용하여, 두 input 이미지가 동일한 클라스인지, 다른 클라스인지를 비교할 수 있습니다.<br/>

### Few Shot Learning
![Few](/assets/fewshot-Kor.png)
SNN을 통해 학습된 클라스 데이터를 바탕으로 유사도를 비교한 모델을 제작합니다.<br/>
이 모델을 이용하여, 검색할 이미지 penguin이 들어왔다고 합니다.<br/>
이 이미지를 기존의 학습된 클라스와 유사도를 비교합니다.<br/>
가장 유사도 점수가 높은 클라스를 반환하여 해당 이미지가 어떤 클라스인지를 예측합니다.<br/>

### 시연 영상
유튜브 링크 업로드로 예정

## 5. 사용법
### Python Requirements
```
python == 3.7.0
numpy == 1.21.6
tensorflow == 2.11.0
picamera2 == 2.6.0
Pillow == 9.5.0
```
본 프로젝트는 Raspberry Pi4 와 Pi Camera2를 이용하여 진행되었습니다.
데스크톱 환경에서 사용 시, 촬영 기능은 제공되지 않습니다.

### 사용 전 설정사항

- verifyClass(file_path) 함수 내의 class_folder를 Black 폴더로 지정해주세요.
- tflite_model_path 경로를 new_model.tflite로 설정해주세요.
- capture_image() 함수 내의 전처리 된 이미지 저장 경로인 file_path를 설정해주세요.
- perform_classification() 함수 내의 image_paths 의 아이콘 경로를 설정해주세요.

### 파일 설명
- GUI.py : Siamese Neural Network로 이미지 비교를 진행하는 심플한 예시 프로그램입니다.
- Siam.py : 메인 프로그램으로 입력 이미지가 어떤 부품인지 판별하는 프로그램 입니다.
- modeler.py : SNN 모델 훈련용 코드 입니다. 한번에 모든 이미지 쌍을 메모리에 탑재하기에 메모리 사용량에 주의해주세요.
- modeler2.py : SNN 모델 훈련용 코드입니다. 제네레이터로 이미지 쌍을 나누어 메모리에 탑재합니다.
- liteTester.py : tflite로 변환된 모델 테스트 코드입니다.
- tflitemaker.py : SNN 모델로 훈련된 saved_model을 float16 방식으로 양자화하여 변환시켜 줍니다.

ESP32 폴더는 ESP32-CAM에 모델 업로드 테스트용 코드입니다. 프로젝트 실행시 사용되지 않습니다. 

### UI 설명

![UI](/assets/UI.png)  

화면 구성은 다음과 같이 3개의 버튼으로 구성되어 있습니다. <br/>
이미지 선택 버튼으로 기존에 촬영된 이미지를 로드할 수 있습니다 <br/>
이미지 촬영 버튼으로 판별을 진행하고자 하는 이미지 촬영이 가능합니다

![load](/assets/load.png)

이미지가 정상적으로 로드되면, 좌측 상단에 이미지가 표시됩니다. <br/>
로드가 완료된 후 판별 진행 버튼을 누르시면 결과가 다음과 같이 나옵니다.

![result](/assets/result.png)

