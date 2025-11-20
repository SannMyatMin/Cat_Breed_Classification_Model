import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class CatBreedPredictor:
    def __init__(self, model_path, class_names):
        self.model  = keras.models.load_model(model_path)
        self.class_names = class_names

    def predict_breed(self, image, top_k=3):
        img = keras.preprocessing.image.load_image(image, target_size=(380,380))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.efficientnet.preprocess_input(img_array)

        predictions = self.model.predict(img_array)

        top_indices     = np.argsort(predictions[0])[-top_k:][::-1]
        top_breeds      = [self.class_names[i] for i in top_indices]
        top_confidences = [predictions[0][i] for i in top_indices]

        return list(zip(top_breeds, top_confidences))
     