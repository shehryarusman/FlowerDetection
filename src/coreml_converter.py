# %% Packages

import json
import tensorflow as tf

# %% Loading models and data

# Model
keras_path = "./models/oxford_flower102_fine_tuning.h5"
keras_model = tf.keras.models.load_model(keras_path)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()
with open("./models/oxford_flower_102.tflite", "wb") as f:
    f.write(tflite_model)

# Labels
labels_path = "./data/cat_to_name.json"
with open(labels_path) as json_file:
    labels_dict = json.load(json_file)
sorted_labels_dict = sorted(labels_dict.items(), key=lambda x: int(x[0]))
label_values = [x[1] for x in sorted_labels_dict]
textfile = open("./models/labels_flowers.txt", "w")
for element in label_values:
    textfile.write(element + "\n")
textfile.close()
