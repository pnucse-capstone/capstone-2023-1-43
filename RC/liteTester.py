import os
import numpy as np
from PIL import Image
import tensorflow as tf
import random

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

# 이미지 크기 설정
input_shape = (480, 640, 3)

# 클래스 데이터 로딩
class_folder = '/Users/NUHEAT/Desktop/Grad/Siamese/class'
images_by_class = load_images_from_folder(class_folder, input_shape)

# 입력 이미지 전처리
input_image_path = '/Users/NUHEAT/Desktop/Grad/Siamese/new/122.jpg'
input_image = Image.open(input_image_path)
input_image = input_image.resize((input_shape[1], input_shape[0]))
input_image_array = np.array(input_image)
input_image_array = input_image_array.astype(np.float32)

# TensorFlow Lite 모델 로딩
tflite_model_path = '/Users/NUHEAT/Desktop/Grad/Siamese/saved_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 각 클래스의 앵커 이미지 선택 및 전처리
class_anchor_images = {}
for class_name, class_images in images_by_class.items():
    anchor_image = random.choice(class_images)
    anchor_image = Image.fromarray(anchor_image.astype('uint8'))
    anchor_image = anchor_image.resize((input_shape[1], input_shape[0]))
    class_anchor_images[class_name] = np.array(anchor_image).astype(np.float32)

# 입력 이미지와 각 클래스의 앵커 이미지 간 유사도 계산
similarities = {}

for class_name, anchor_image in class_anchor_images.items():
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_image_array, axis=0))
    interpreter.set_tensor(input_details[1]['index'], np.expand_dims(anchor_image, axis=0))
    interpreter.invoke()

    similarity_score = interpreter.get_tensor(output_details[0]['index'])
    similarities[class_name] = similarity_score

# 유사도가 가장 높은 클래스 예측
predicted_class = max(similarities, key=similarities.get)
similarity_score = similarities[predicted_class]

print(f"The input image is most similar to class: {predicted_class}")
print(f"Similarity Score: {similarity_score[0][0]}")
