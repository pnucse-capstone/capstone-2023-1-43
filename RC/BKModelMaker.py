import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import random


# Siamese Network 모델 정의
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


# 모델 컴파일
input_shape = (96, 96, 1)
siamese_model = build_siamese_model(input_shape)
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# 데이터 준비
def load_images_from_folder(folder):
    images_by_class = {}  # 클래스별 이미지 리스트를 저장할 딕셔너리

    for class_name in os.listdir(folder):
        if class_name.startswith('.'):
            continue  # .DS_Store와 같은 시스템 파일은 무시

        class_folder = os.path.join(folder, class_name)
        images = []

        for filename in os.listdir(class_folder):
            if filename.startswith('.'):
                continue  # .DS_Store와 같은 시스템 파일은 무시

            img = Image.open(os.path.join(class_folder, filename))
            if img is not None:
                img = img.resize((input_shape[1], input_shape[0]))  # 이미지 크기 조정
                img_array = np.array(img)
                img_array = img_array.astype(np.float32)
                images.append(img_array)

        images_by_class[class_name] = images

    return images_by_class


# 이미지 데이터 경로 설정
class_folder = '/Users/NUHEAT/Desktop/Grad/Siamese/Black'

# 클래스 데이터 로딩
images_by_class = load_images_from_folder(class_folder)

# 이미지 쌍 생성
image_pairs = []
labels = []

for class_name, class_images in images_by_class.items():
    for i, anchor_image in enumerate(class_images):
        for positive_image in class_images[i + 1:]:  # 현재 클래스 내에서 다른 이미지들과 조합
            # 같은 클래스 이미지 조합
            image_pairs.append((anchor_image, positive_image))
            labels.append(1)  # 같은 클래스에서 온 이미지는 유사함을 나타냄

            # 다른 클래스 이미지 조합
            random_class_name = random.choice(list(images_by_class.keys()))
            while random_class_name == class_name:  # 다른 클래스 선택 반복
                random_class_name = random.choice(list(images_by_class.keys()))
            random_class_images = images_by_class[random_class_name]
            negative_image = random.choice(random_class_images)  # 무작위 클래스에서 이미지 선택
            image_pairs.append((anchor_image, negative_image))
            labels.append(0)  # 다른 클래스에서 온 이미지는 유사하지 않음을 나타냄

image_pairs = np.array(image_pairs)
labels = np.array(labels)

# 모델 훈련
siamese_model.fit([image_pairs[:, 0], image_pairs[:, 1]], labels, epochs=10, batch_size=32) #배치 32에서 16으로 변경함 정확도 5942 더 떨어진듯?

# 훈련된 모델 저장
saved_model_path = '/Users/NUHEAT/Desktop/Grad/Siamese/new_model'
os.makedirs(saved_model_path, exist_ok=True)  # 폴더가 없으면 생성

siamese_model.save(saved_model_path)
print(f"Trained model saved at {saved_model_path}")
