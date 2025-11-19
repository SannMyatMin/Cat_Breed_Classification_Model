import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

class CatBreedClassifier:
    def __init__(self, num_classes, model_size='medium', img_size=(224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_size = model_size
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build the model with transfer learning"""
        # Select base model based on data size
        if self.model_size == 'small':
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif self.model_size == 'medium':
            base_model = keras.applications.EfficientNetB2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:  # large
            base_model = keras.applications.EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=(380, 380, 3)  # B4 uses larger input
            )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build custom classifier
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Preprocess input for EfficientNet
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        
        # Advanced classifier head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        print(f"Built model with {base_model.name}")
        return self.model
    
    def compile_model(self, initial_learning_rate=1e-3):
        """Compile the model with appropriate settings"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print("Model compiled successfully!")
    
    def get_callbacks(self):
        """Define training callbacks"""
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                'best_cat_breed_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        return callbacks
