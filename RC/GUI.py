import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 파일 선택 버튼 클릭 시 실행될 함수
def open_file_dialog_1():
    global image_path1  # 이미지 경로를 전역 변수로 지정
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.gif *.bmp *.ppm *.pgm")])
    if file_path:
        image_path1 = file_path  # 이미지 경로를 저장
        display_image(1, file_path)  # 이미지를 표시

def open_file_dialog_2():
    global image_path2  # 이미지 경로를 전역 변수로 지정
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.gif *.bmp *.ppm *.pgm")])
    if file_path:
        image_path2 = file_path  # 이미지 경로를 저장
        display_image(2, file_path)  # 이미지를 표시

# 이미지를 보여주는 함수
def display_image(frame_num, file_path):
    image = Image.open(file_path)
    image = resize_image(image, (600, 450))  # 이미지 크기를 600x450으로 리사이즈
    photo = ImageTk.PhotoImage(image)

    # 이미지를 캔버스에 표시하면서 상단 마진을 추가
    if frame_num == 1:
        canvas_1.create_image(0, 10, anchor="nw", image=photo)  # Y 좌표를 10으로 설정
        canvas_1.image = photo  # 이미지를 레퍼런스로 유지
    else:
        canvas_2.create_image(0, 10, anchor="nw", image=photo)  # Y 좌표를 10으로 설정
        canvas_2.image = photo  # 이미지를 레퍼런스로 유지

# 이미지 리사이즈 함수
def resize_image(image, size):
    return image.resize(size)

# 이미지 비교 함수

def compare_images():
    if image_path1 and image_path2:  # 두 이미지 경로가 존재하는 경우에만 계산
        similarity_percentage = calculate_similarity(image_path1, image_path2)
        result_label.config(text=f"두 이미지의 유사도: {similarity_percentage:.2f}%")  # 결과를 레이블에 업데이트



"""
# 이미지 비교 함수
def compare_images():
    # 비교 결과를 저장할 리스트 생성
    similarity_scores = []

    if image_path1 and image_path2:  # 두 이미지 경로가 존재하는 경우에만 계산
        num_iterations = 100  # 원하는 반복 횟수를 설정하세요

        for _ in range(num_iterations):
            similarity_percentage = calculate_similarity(image_path1, image_path2)
            similarity_scores.append(similarity_percentage)

        # similarity_scores 리스트를 정렬
        sorted_similarity_scores = sorted(similarity_scores)

        # 중앙값 계산
        if num_iterations % 2 == 0:
            # 짝수 개의 값이면 중앙 두 값의 평균을 사용
            median = (sorted_similarity_scores[num_iterations // 2 - 1] + sorted_similarity_scores[
                num_iterations // 2]) / 2
        else:
            # 홀수 개의 값이면 중앙값을 사용
            median = sorted_similarity_scores[num_iterations // 2]

        result_label.config(text=f"두 이미지의 중앙값: {median:.2f}")

"""

# Tkinter 창 생성
root = tk.Tk()
root.title("이미지 유사도 비교기")

# 창 크기를 1280x720 픽셀로 고정
root.geometry("1260x720")

# 파일 선택 버튼 생성
open_button_1 = tk.Button(root, text="파일 선택 1", command=open_file_dialog_1, width=15, height=2)
open_button_2 = tk.Button(root, text="파일 선택 2", command=open_file_dialog_2, width=15, height=2)
compare_button = tk.Button(root, text="비교하기", command=compare_images, width=15, height=2)

# 캔버스를 생성하여 이미지를 표시할 프레임 생성 (크기를 600x450으로 조정)
canvas_1 = tk.Canvas(root, width=600, height=450)
canvas_2 = tk.Canvas(root, width=600, height=450)

# 버튼과 캔버스를 그리드로 배치
canvas_1.grid(row=1, column=1, padx=10, pady=10)  # padx 및 pady를 사용하여 간격을 조절
canvas_2.grid(row=1, column=2, padx=10, pady=10)  # padx 및 pady를 사용하여 간격을 조절

open_button_1.grid(row=2, column=1, pady=20)  # 파일 선택 1 버튼 아래에 20픽셀의 간격 추가
open_button_2.grid(row=2, column=2, pady=20)  # 파일 선택 2 버튼 아래에 20픽셀의 간격 추가
compare_button.grid(row=3, column=1, columnspan=2)

# 이미지 파일 경로를 저장할 전역 변수 초기화
image_path1 = None
image_path2 = None

# 비교 결과를 표시할 레이블 생성
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.grid(row=4, column=1, columnspan=2, pady=10)  # 레이블을 그리드로 배치



def build_siamese_model(input_shape):
    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)

    base_network = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten()
    ])

    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)

    # 두 이미지 임베딩 간의 거리 계산
    distance = tf.keras.layers.Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]))([encoded_a, encoded_b])

    # 유사도 예측
    similarity_prediction = layers.Dense(1, activation='sigmoid')(distance)

    siamese_model = tf.keras.Model(inputs=[input_a, input_b], outputs=similarity_prediction)

    return siamese_model


def calculate_similarity(image_path1, image_path2):
    # 이미지를 모델 입력 형식에 맞게 전처리
    input_shape = (640, 480)  # 높이와 너비를 직접 지정
    siamese_model = build_siamese_model((640, 480, 3))
    image1 = load_img(image_path1, target_size=input_shape)  # 이미지를 읽어와서 크기 조정
    image2 = load_img(image_path2, target_size=input_shape)  # 이미지를 읽어와서 크기 조정
    image1 = img_to_array(image1)  # 이미지를 NumPy 배열로 변환
    image2 = img_to_array(image2)  # 이미지를 NumPy 배열로 변환
    image1 = np.expand_dims(image1, axis=0)  # 배치 차원 추가
    image2 = np.expand_dims(image2, axis=0)  # 배치 차원 추가

    # 두 이미지 간의 유사도 예측
    similarity_score = siamese_model.predict([image1, image2])

    # 유사도를 퍼센트로 변환하여 반환
    similarity_percentage = similarity_score[0][0] * 100
    return similarity_percentage

# 모델 생성
input_shape = (640, 480, 3)  # 입력 이미지 크기

# Tkinter 루프 시작
root.mainloop()
