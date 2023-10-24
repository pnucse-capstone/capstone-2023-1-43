import os
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import random
from picamera2 import Picamera2

def load_images_from_folder(folder, input_shape):
    images_by_class = {}

    for class_name in os.listdir(folder):
        if class_name.startswith('.'):
            continue

        class_folder = os.path.join(folder, class_name)
        images = []

        for filename in os.listdir(class_folder):
            if filename.startswith('.'):
                continue

            img = Image.open(os.path.join(class_folder, filename))
            if img is not None:
                img = img.resize((input_shape[1], input_shape[0]))
                img_array = np.array(img)
                img_array = img_array.astype(np.float32)
                images.append(img_array)

        images_by_class[class_name] = images

    return images_by_class

#인풋 사이즈, 클라스 폴더, 경로를 파라미터로 이용?

def verifyClass(file_path):
    # 이미지 크기 설정
    input_shape = (96, 96, 1)

    # 클래스 데이터 로딩
    class_folder = '/Users/NUHEAT/Desktop/Grad/Siamese/Black'
    images_by_class = load_images_from_folder(class_folder, input_shape)

    # 입력 이미지 전처리
    input_image_path = file_path
    input_image = Image.open(input_image_path)
    input_image = input_image.resize((input_shape[1], input_shape[0]))
    input_image_array = np.array(input_image)
    input_image_array = input_image_array.astype(np.float32)

    input_image_array = input_image_array.reshape((1, input_shape[0], input_shape[1], 1))

    # TensorFlow Lite 모델 로딩
    tflite_model_path = '/Users/NUHEAT/Desktop/Grad/Siamese/new_model.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 예측 결과를 저장할 딕셔너리 초기화
    predictions_count = {}

    # 예측을 10번 수행하고 결과를 저장
    for _ in range(10):
        class_anchor_images = {}
        for class_name, class_images in images_by_class.items():
            anchor_image = random.choice(class_images)
            anchor_image = Image.fromarray(anchor_image.astype('uint8'))
            anchor_image = anchor_image.resize((input_shape[1], input_shape[0]))
            anchor_image = np.array(anchor_image).astype(np.float32)
            anchor_image = anchor_image.reshape((1, input_shape[0], input_shape[1], 1))
            class_anchor_images[class_name] = anchor_image

        similarities = {}
        for class_name, anchor_image in class_anchor_images.items():
            interpreter.set_tensor(input_details[0]['index'], input_image_array)
            interpreter.set_tensor(input_details[1]['index'], anchor_image)
            interpreter.invoke()
            similarity_score = interpreter.get_tensor(output_details[0]['index'])
            similarities[class_name] = similarity_score

        predicted_class = max(similarities, key=similarities.get)

        # 예측 결과를 딕셔너리에 저장
        if predicted_class in predictions_count:
            predictions_count[predicted_class] += 1
        else:
            predictions_count[predicted_class] = 1

    # 가장 많이 나온 클래스 찾기
    most_common_class = max(predictions_count, key=predictions_count.get)

    return most_common_class


# 전역 변수로 image_label 및 result_label 선언
image_label = None
result_label = None
result_frame = None
file_path = None

def select_image():
    global image_label, file_path  # 전역 변수로 사용

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.gif *.bmp")])
    if file_path:
#        print(f"선택한 이미지 경로: {file_path}")
        if image_label:
            image_label.destroy()  # 이미지 프레임에 있는 이전 이미지 삭제
        display_image(file_path)

def display_image(file_path):
    global image_label  # 전역 변수로 사용

    image = Image.open(file_path)
    image = image.resize((200, 200), Image.LANCZOS)  # 이미지 크기 조정
    photo = ImageTk.PhotoImage(image)

    # 이미지 프레임에 이미지 표시 및 중앙 정렬
    image_label = tk.Label(image_frame, image=photo)
    image_label.image = photo
    image_label.grid(row=0, column=0, padx=(image_frame.winfo_width() - 200) // 2, pady=(image_frame.winfo_height() - 200) // 2)

def capture_image():
    # 이미지 촬영 후 특정 경로에 파일 저장, 그 파일 경로만 토스해주면 됨.
    # file_path만 잡아주고 display_image 함수 이용하면 될 것으로 보임
    # CV2로 흑백에 크기 96*96 조절부 넣어주는 걸 여기서 처리해야 될 것.
    global file_path
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (2464, 2464)})  # Max 3280*2464
    picam2.configure(camera_config)
    picam2.start()

    filename = "captured_image.jpg"
    picam2.capture_file(filename)
    picam2.close()

    image = Image.open(filename)
    image = image.resize((96, 96))
    image = image.convert('L')
    file_path = "파일 경로 지정"

    image.save(file_path)
    display_image(file_path)


def perform_classification():
    global result_frame, result_label, file_path  # 전역 변수로 사용

    most_similar_class = verifyClass(file_path) # 모델 판정 함수
    result_name = tk.Label(window, text=f"{most_similar_class}", font=("Helvetica", 30), fg="black", bg="white")
    result_name.grid(row=5, column=0, padx=20, pady=20, columnspan=2, sticky="nsew")  # Center-align the label

    image_paths = {
        "Speaker": '/Users/NUHEAT/Desktop/Grad/Siamese/ICON/Speaker.png',
        "RearCam": '/Users/NUHEAT/Desktop/Grad/Siamese/ICON/Cam.png',
        "Dock": '/Users/NUHEAT/Desktop/Grad/Siamese/ICON/Dock.png',
        "Vibrator": '/Users/NUHEAT/Desktop/Grad/Siamese/ICON/Vib.png',
        "EarSpeaker": '/Users/NUHEAT/Desktop/Grad/Siamese/ICON/Ear.png',
        # 라즈베리파이에서 경로 재설정 필요
    }

    # most_similar_class 값에 따라 이미지 경로 선택
    image_path = image_paths.get(most_similar_class, '/Users/NUHEAT/Desktop/Grad/Siamese/ICON/None.png')

    # 선택한 이미지 열기
    image = Image.open(image_path)
    image = image.resize((200, 200), Image.LANCZOS)  # 이미지 크기 조정
    photo = ImageTk.PhotoImage(image)

    # 이미지 프레임에 이미지 표시 및 중앙 정렬
    image_label = tk.Label(result_frame, image=photo)
    image_label.image = photo
    image_label.grid(row=4, column=0,columnspan=2)

    explain = {'EarSpeaker' : '통화용 소리를 발생시키며\n조도 센서가 동반되기도 합니다.',
               "RearCam" : "촬영용 카메라로 보통은 둘 이상의 \n 망원, 광각 카메라로 구성됩니다.",
               "Dock" : "충전 단자가 달린 부품으로 \n 마이크와 스피커 단자가 동반됩니다.",
               "Speaker" : "소리를 내는 주된 스피커로 \n 정식 명칭은 라우드 스피커입니다",
               "Vibrator" : "진동을 내는 부품으로 \n 탭틱 엔진이라고도 불립니다."}


    result_text = tk.Label(window, text=f"{explain[most_similar_class]}", font=("Helvetica", 16), fg="black",
                           bg="white")
    result_text.grid(row=6, column=0, padx=20, pady=20, columnspan=2, sticky="nsew")  # Center-align the label



# 윈도우 생성
window = tk.Tk()
window.title("이미지 판정 앱")
window.geometry("400x1000")

# 이미지 프레임 생성
image_frame = tk.Frame(window, width=200, height=200)
image_frame.grid(row=0, column=0, padx=20, pady=20, rowspan=2)

# 이미지 프레임 크기를 고정하고 크기 조정 비활성화
image_frame.grid_propagate(False)

# 이미지 선택 버튼 생성 (크기 조절)
select_image_button = tk.Button(window, text="이미지 선택", command=select_image, width=10, height=2)
select_image_button.grid(row=0, column=1, padx=20, pady=10)

# 이미지 촬영 버튼 생성 (크기 조절)
capture_image_button = tk.Button(window, text="이미지 촬영", command=capture_image, width=10, height=2)
capture_image_button.grid(row=1, column=1, padx=20, pady=10)

# 판정 진행 버튼 생성 (크기 조절)
process_button = tk.Button(window, text="판정 진행", command=perform_classification, width=40, height=2)
process_button.grid(row=3, column=0, padx=20, pady=10, columnspan=2)


# "result frame" 프레임 생성
result_frame = tk.Frame(window, width=205, height=205, bg="grey")
result_frame.grid(row=4, column=0, padx=20, pady=20, columnspan=2)

# "result frame" 크기를 고정하고 크기 조정 비활성화
result_frame.grid_propagate(False)




# GUI 실행
window.mainloop()
