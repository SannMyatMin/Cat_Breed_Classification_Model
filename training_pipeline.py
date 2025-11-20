import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class TrainingPipeline:
    def __init__(self, model, train_gen, val_gen):
        self.model     = model
        self.train_gen = train_gen
        self.val_gen   = val_gen
        self.history   = None

    def train_initial(self, epochs=15):
        print("Stage-1: Training with frozen base model")
        history1 = self.model.fit(
            self.train_gen,
            epochs = epochs,
            validation_data = self.val_gen,
            callbacks = self.model.get_claabacks(),
            verbose = 1 )
        
        return history1
    
    def unfreeze_and_fine_tune(self, fine_tune_epochs=20, fine_tune_lr=1e-5):
        print("Stage-2: Fine-tuning base model(unfrozen layrs)")
        base_model = self.model.layers[2]
        base_model.trainable = True

        fine_tune_at = len(base_model.layers) // 2
        for layers in base_model.layers[:fine_tune_at]:
            layers.trainable = True
        print(f"Unfroze {len(base_model.layers) - fine_tune_at} layers for fine_tuning")

        # Recompile model for unfrozen layers
        self.model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=fine_tune_lr/10),
            loss = "categorical_crossentropy",
            metrics = ["accuracy",
                       keras.metrics.Precision(name = "precision"),
                       keras.metrics.Recall(name = "recall") ])
        
        history2 = self.model.fit(
            self.train_gen,
            epochs = fine_tune_epochs,
            initial_epoch = self.history.epoch[-1] + 1 if self.history else 0,
            validation_gen = self.val_gen,
            verbose = 1 )
        
        return history2
    
    def train_model(self, initial_epochs=15, fine_tune_epochs=20):
        history1 = self.train_initial(epochs = initial_epochs)
        history2 = self.unfreeze_and_fine_tune(fine_tune_epochs = fine_tune_epochs)
        self.history = self._combine_history(history1, history2)

        return self.history
        
    def _combine_history(self, history1, history2):
        combined_history = {}
        for key in history1.history.keys():
            combined_history[key] = history1.history[key] + history2.history[key]
        return combined_history 
    
    def plot_training_history(self, history):
   
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Training Loss')
        axes[0, 1].plot(history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history['precision'], label='Training Precision')
        axes[1, 0].plot(history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(history['recall'], label='Training Recall')
        axes[1, 1].plot(history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

