# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import regularizers
# from tensorflow.keras.models import load_model

# MODEL_PATH = "models\mobilenet_mango_model.h5"
# TFLITE_PATH = "models\mobilenet_mango_model.tflite"

# num_classes = 5
# # loading the model
# base_model = tf.keras.applications.MobileNetV2(input_shape=(150,150,3),
#                          include_top=False,  # exclude the original FC layers
#                          weights='imagenet')
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.5)(x)   # helps reduce overfitting
# predictions = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001))(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# model.load_weights(MODEL_PATH)

# # converting the model using tflite

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()


# # saving the tflite model
# with open(TFLITE_PATH, 'wb') as f:
#     f.write(tflite_model)

# print(f"Model converted and saved to {TFLITE_PATH}")

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

MODEL_PATH = "D:/FYP/models/mobilenet_mango_model.h5"
TFLITE_PATH = "D:/FYP/models/mobilenet_mango_model_compat.tflite"

num_classes = 5

# Rebuild the same architecture
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001))(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Load weights only (not architecture)
model.load_weights(MODEL_PATH)
print("Weights loaded successfully")

# Convert with compatibility settings
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_types = [tf.float32]

tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Saved: {len(tflite_model)/1024/1024:.1f} MB")