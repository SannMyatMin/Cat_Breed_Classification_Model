import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class CatBreedPredictor:
    def __init__(self, model_path, class_names):
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names
    
    def predict_breed(self, image_path, top_k=3):
        """Predict cat breed from image"""
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.efficientnet.preprocess_input(img_array)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        
        # Get top K predictions
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_breeds = [self.class_names[i] for i in top_indices]
        top_confidences = [predictions[0][i] for i in top_indices]
        
        return list(zip(top_breeds, top_confidences))
    
    def predict_and_visualize(self, image_path):
        """Predict and visualize results"""
        predictions = self.predict_breed(image_path)
        
        # Display image and predictions
        img = keras.preprocessing.image.load_img(image_path)
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input Image')
        
        plt.subplot(1, 2, 2)
        breeds = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(breeds)))
        bars = plt.barh(breeds, confidences, color=colors)
        plt.xlabel('Confidence')
        plt.title('Breed Predictions')
        plt.xlim(0, 1)
        
        # Add confidence values on bars
        for bar, confidence in zip(bars, confidences):
            plt.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2,
                    f'{confidence:.2%}', ha='right', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return predictions