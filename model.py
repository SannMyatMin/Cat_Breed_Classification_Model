import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

class CatBreedClassifierAI:
    def __init__(self, num_classes, model_size, img_size):
        self.num_classes = num_classes
        self.model_size  = model_size
        self.image_size  = img_size
        self.model       = None
        
    def build_model(self):
        if self.model_size == "medium":
            base_model = keras.applications.EfficientNetB2(
                weights = "imagenet",
                include_top = False,
                input_shape =(*self.image_size, 3) )
        else:
            base_model = keras.applications.EfficientNetB4(
                weights = "imagenet",
                include_top = False,
                input_shape = (*self.image_size, 3) )
        base_model.trainable = False

        inputs = keras.Input(shape=(*self.image_size, 3))
        # Preprocess image
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)

        # Classifier Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.2)(x) 

        # Output Layer
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def compile_model(self, initial_learning_rate=1e-3):
        self.model.compile(
            optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate),
            loss = "categorical_crossentropy",
            metrics = ["accuracy",
                       keras.metrics.Precision(name="precision"),
                       keras.metrics.Recall(name="recall") ])
        print("Model is compiled successsfully")

    def get_callbacks(self):
        callbacks = [
            # Save Best Model
            keras.callbacks.ModelCheckpoint(
                "best_cat_breed_classification_model.h5",
                monitor = "val_accuracy",
                save_best_only = True,
                mode = "max",
                verbose = 1 ),

            # Early Stopping
            keras.callbacks.EarlyStopping(
                monitor = "val_loss",
                patience = 10,
                restore_best_weights = True,
                verbose = 1 ),

            # Reduce Learning-Rate on Plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor = "val_loss",
                factor = 0.2,
                min_lr = 1e-7,
                verbose = 1 ),

            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir = "./logs",
                histogram_freq = 1 )
        ]

        return callbacks