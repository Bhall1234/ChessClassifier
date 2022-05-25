# Code extended code from https://www.tensorflow.org/tutorials/images/classification
# Code from workshop 1 of CMP2020 - 2122, credit to original owner. Some changes made to adapt to the teams project.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
import sys
import PIL
import tensorflow as tf

import pathlib
import shutil

from tensorflow import keras
#from tensorflow import confusion_matrix
from keras import layers
from keras.models import Sequential

# Confusion Matrix Storage
store_probability = [] 
store_labels = [0,1,2,3,4,5]

exec_mode = sys.argv[1] #train|test
data_dir = pathlib.Path("C:\\Users\\benha\\Documents\\GitHub\\Chess_Classifier\\test")
image_count = len(list(data_dir.glob('*/*.png')))
print("|images|="+str(image_count))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(5000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

#model.save("chess_classifier.h5")

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if exec_mode == 'train':
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
  epochs=100
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[cp_callback]
  )

  model.save("chess_classifier_video.h5")

elif exec_mode == 'test':
  latest = tf.train.latest_checkpoint(checkpoint_dir)
  model.load_weights(latest)
 
  test_data_dir = "C:\\Users\\benha\\Documents\\GitHub\\Chess_Classifier\\dataset"  
  pieces = os.listdir(test_data_dir)
  for piece in pieces:
    filePath = test_data_dir+"/"+ piece
  
    files = os.listdir(filePath)
    for fileName in files:
  
      if fileName.endswith(".png") or fileName.endswith(".jpg"): # or .jpg!!!
        chess_path = filePath+"/"+fileName
        img = tf.keras.utils.load_img(chess_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        scores = tf.nn.softmax(predictions[0]) # Softmax activation function
        #store_probability.append(scores) # store probability for use in confusion matrix
        predicted_class = class_names[np.argmax(scores)]
        print(str(chess_path)+" class="+predicted_class+" prob.="+str(np.max(scores)))
    print("---")

    #print(confusion)
  #confusion = tf.math.confusion_matrix(labels=store_labels, predictions=scores)
  #sess = tf.Session()
  #print(confusion.eval(session=sess))

else:
  print("UNKNOWN: exec_mode="+str(exec_mode))

  #python ChessClassifier.py train
  #python ChessClassifier.py test