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
input_shape = (320, 240, 1)
#흑백이라 채널을 1로 변경, 크기가 320*240 -> 96*96

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

# 데이터 제너레이터 생성
def data_generator(images_by_class, batch_size=32):
    while True:
        image_pairs = []
        labels = []

        for class_name, class_images in images_by_class.items():
            for i, anchor_image in enumerate(class_images):
                for positive_image in class_images[i + 1:]:
                    image_pairs.append((anchor_image, positive_image))
                    labels.append(1)

                    random_class_name = random.choice(list(images_by_class.keys()))
                    while random_class_name == class_name:
                        random_class_name = random.choice(list(images_by_class.keys()))
                    random_class_images = images_by_class[random_class_name]
                    negative_image = random.choice(random_class_images)
                    image_pairs.append((anchor_image, negative_image))
                    labels.append(0)

                    if len(image_pairs) >= batch_size:
                        yield [np.array(image_pairs)[:, 0], np.array(image_pairs)[:, 1]], np.array(labels)
                        image_pairs = []
                        labels = []

# 이미지 데이터 경로 설정
class_folder = '/Users/NUHEAT/Desktop/Grad/Siamese/class'

# 클래스 데이터 로딩
images_by_class = load_images_from_folder(class_folder)

# 모델 훈련
batch_size = 8
steps_per_epoch = len(images_by_class) * len(images_by_class[list(images_by_class.keys())[0]]) // batch_size
siamese_model.fit(data_generator(images_by_class, batch_size), steps_per_epoch=steps_per_epoch, epochs=10)

# 훈련된 모델 저장
saved_model_path = '/Users/NUHEAT/Desktop/Grad/Siamese/saved_model'
os.makedirs(saved_model_path, exist_ok=True)

siamese_model.save(saved_model_path)
print(f"Trained model saved at {saved_model_path}")

