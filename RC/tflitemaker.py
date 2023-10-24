import tensorflow as tf

# Load the SavedModel from .pb file
model = tf.saved_model.load('/Users/NUHEAT/Desktop/Grad/Siamese/saved_model/')



# Convert the SavedModel to a TFLite model with quantization
converter = tf.lite.TFLiteConverter.from_saved_model('/Users/NUHEAT/Desktop/Grad/Siamese/saved_model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]


converter.target_spec.supported_types = [tf.float16]
#converter.target_spec.supported_types = [tf.uint8]


tflite_model = converter.convert()

with open('saved_model.tflite', 'wb') as f:
    f.write(tflite_model)
