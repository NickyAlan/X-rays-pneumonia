from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os

def preprocessing(image_path, image_size = (224,224)) :
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1) # grayscale 
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # normalize scale between 0 to 1
    image = tf.image.resize(image, size=image_size)
    return image

def creat_dataset_batch(X, y=None, batch_size=32) :
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data = data.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return data

if __name__ == '__main__' :
    images_path = os.listdir('predicts_custom')
    images_filepath = [os.path.join('predicts_custom', imagepath) for imagepath in images_path]

    file_types = []
    for file in images_filepath :
        file_type = file.split('.')[-1]
        if file_type != 'jpeg' : 
            print(' -- please use jpeg file format !') 
            sys.exit()
        
    BATCH_SIZE = 8
    predict_data = creat_dataset_batch(images_filepath, batch_size=BATCH_SIZE)

    class_names = ['NORMAL', 'PNEUMONIA']
    model = keras.models.load_model('save_model/x-ray-94acc.h5')
    proba_1 = model.predict(predict_data).reshape(-1,)
    pred_class = np.array([1 if prob > 0.4416384 else 0 for prob in proba_1]).reshape(-1,)
    
    for i in range(len(pred_class)) :
        print(images_filepath[i])
        print(class_names[pred_class[i]])
        proba = proba_1[i] if pred_class[i] == 1 else np.abs(1-proba_1[i])
        print(proba)
        print('--\n')


    
