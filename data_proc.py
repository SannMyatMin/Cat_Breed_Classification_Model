import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

class CatDataPreprocessor:
    def __init__(self, dataset_dir, csv_filename='cat_breed.csv', images_dir='images', 
                 img_size=(224, 224), batch_size=32, validation_split=0.2, test_split=0.1):
        self.dataset_dir = Path(dataset_dir)
        self.csv_path = self.dataset_dir / csv_filename
        self.images_dir = self.dataset_dir / images_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.df = None
        self.class_names = []
        
    def create_image_dataframe(self):
        """Create dataframe from images folder structure (as in your code)"""
        print("Creating image dataframe from folder structure...")
        
        image_data = []
        base_dir = Path('dataset/images')

        for breed_folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, breed_folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith((".jpg", ".png", ".jpeg")):
                        name_without_extension = os.path.splitext(filename)[0]
                        img_id = name_without_extension.split("_")[0]  # Extract ID before underscore
                        image_data.append({
                            "img_path": os.path.join(folder_path, filename),
                            "id": img_id,
                            "breed": breed_folder  # Use folder name as breed
                        })
        
        img_df = pd.DataFrame(image_data)
        print(f"Created image dataframe with {len(img_df)} samples")
        return img_df
    
    def merge_with_csv(self, img_df):
        """Merge image dataframe with CSV data"""
        print("Loading and merging with CSV...")
        
        # Load CSV
        csv_df = pd.read_csv(self.csv_path)
        print(f"CSV shape: {csv_df.shape}")
        print(f"CSV columns: {csv_df.columns.tolist()}")
        
        # Check if 'breed' column exists in CSV
        if 'breed' not in csv_df.columns:
            raise ValueError("CSV must contain 'breed' column")
        
        # Find ID column in CSV
        id_column = None
        for col in ['id', 'image_id', 'filename']:
            if col in csv_df.columns:
                id_column = col
                break
        
        if id_column is None:
            # If no explicit ID column, assume first column is ID
            id_column = csv_df.columns[0]
            print(f"No ID column found, using '{id_column}' as ID")
        
        print(f"Using '{id_column}' as ID column from CSV")
        
        # Convert ID to string for both dataframes
        csv_df[id_column] = csv_df[id_column].astype(str)
        img_df['id'] = img_df['id'].astype(str)
        
        # Merge on ID
        merged_df = pd.merge(
            img_df, 
            csv_df[[id_column, 'breed']], 
            left_on='id', 
            right_on=id_column, 
            how='inner',
            suffixes=('_from_folder', '_from_csv')
        )
        
        print(f"After merge: {len(merged_df)} samples")
        
        # Check for breed consistency
        breed_mismatch = merged_df[merged_df['breed_from_folder'] != merged_df['breed_from_csv']]
        if len(breed_mismatch) > 0:
            print(f"Warning: {len(breed_mismatch)} breed mismatches between folder and CSV")
            # Use CSV breed as source of truth
            merged_df = merged_df.drop('breed_from_folder', axis=1)
            merged_df = merged_df.rename(columns={'breed_from_csv': 'breed'})
        else:
            merged_df = merged_df.drop('breed_from_csv', axis=1)
            merged_df = merged_df.rename(columns={'breed_from_folder': 'breed'})
        
        return merged_df
    
    def load_and_validate_data(self):
        """Main method to load and validate all data"""
        print("Loading and validating dataset...")
        
        # Step 1: Create image dataframe from folder structure
        img_df = self.create_image_dataframe()
        
        # Step 2: Merge with CSV
        self.df = self.merge_with_csv(img_df)
        
        # Step 3: Validate image files exist
        self._validate_image_files()
        
        # Step 4: Get class names
        self.class_names = sorted(self.df['breed'].unique())
        
        print(f"Final dataset: {len(self.df)} samples across {len(self.class_names)} breeds")
        print(f"Breeds: {self.class_names}")
        
        return self.df
    
    def _validate_image_files(self):
        """Validate that all image paths exist"""
        print("Validating image files...")
        
        missing_files = []
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            if os.path.exists(row['img_path']):
                valid_indices.append(idx)
            else:
                missing_files.append(row['img_path'])
        
        # Keep only valid files
        self.df = self.df.loc[valid_indices]
        
        if missing_files:
            print(f"Warning: {len(missing_files)} image files are missing")
            if len(missing_files) <= 10:
                for path in missing_files:
                    print(f"  Missing: {path}")
        
        print(f"Valid images: {len(self.df)}")
    
    def create_train_val_test_split(self):
        """Create train, validation, and test splits"""
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            self.df, 
            test_size=self.test_split,
            stratify=self.df['breed'],
            random_state=42
        )
        
        # Second split: separate validation from train
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.validation_split,
            stratify=train_val_df['breed'],
            random_state=42
        )
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_data_generators(self, train_df, val_df, test_df=None):
        """Create data generators for training, validation, and testing"""
        
        # Data augmentation for training
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            shear_range=0.15,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        val_test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='img_path',  # Use the full image path we created
            y_col='breed',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = val_test_datagen.flow_from_dataframe(
            val_df,
            x_col='img_path',
            y_col='breed',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = None
        if test_df is not None:
            test_generator = val_test_datagen.flow_from_dataframe(
                test_df,
                x_col='img_path',
                y_col='breed',
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            print(f"Test samples: {test_generator.samples}")
        
        # Update class names from generator
        self.class_names = list(train_generator.class_indices.keys())
        self.class_indices = train_generator.class_indices
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        if test_generator:
            print(f"Test samples: {test_generator.samples}")
        
        return train_generator, validation_generator, test_generator
    
    def analyze_class_distribution(self, train_df, val_df, test_df=None):
        """Analyze and display class distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Train distribution
        train_counts = train_df['breed'].value_counts()
        axes[0, 0].bar(range(len(train_counts)), train_counts.values)
        axes[0, 0].set_title('Training Set Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(range(len(train_counts)))
        axes[0, 0].set_xticklabels(train_counts.index, rotation=90)
        
        # Validation distribution
        val_counts = val_df['breed'].value_counts()
        axes[0, 1].bar(range(len(val_counts)), val_counts.values)
        axes[0, 1].set_title('Validation Set Distribution')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(range(len(val_counts)))
        axes[0, 1].set_xticklabels(val_counts.index, rotation=90)
        
        # Test distribution (if available)
        if test_df is not None:
            test_counts = test_df['breed'].value_counts()
            axes[1, 0].bar(range(len(test_counts)), test_counts.values)
            axes[1, 0].set_title('Test Set Distribution')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(len(test_counts)))
            axes[1, 0].set_xticklabels(test_counts.index, rotation=90)
        
        # Overall statistics
        axes[1, 1].text(0.1, 0.9, f"Total Breeds: {len(self.class_names)}", fontsize=12)
        axes[1, 1].text(0.1, 0.8, f"Training Samples: {len(train_df)}", fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"Validation Samples: {len(val_df)}", fontsize=12)
        if test_df is not None:
            axes[1, 1].text(0.1, 0.6, f"Test Samples: {len(test_df)}", fontsize=12)
        
        axes[1, 1].text(0.1, 0.5, "Class Distribution Stats:", fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.4, f"Min samples per class: {min(train_counts)}", fontsize=10)
        axes[1, 1].text(0.1, 0.3, f"Max samples per class: {max(train_counts)}", fontsize=10)
        axes[1, 1].text(0.1, 0.2, f"Avg samples per class: {train_counts.mean():.1f}", fontsize=10)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return train_counts, val_counts
    
    

# Simplified usage for your specific case
def setup_data_pipeline():
    """Setup complete data pipeline for your dataset structure"""
    
    # Configuration
    DATASET_DIR = "dataset"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # Initialize preprocessor
    preprocessor = CatDataPreprocessor(
        dataset_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Load and validate data
    df = preprocessor.load_and_validate_data()
    
    # Create splits
    train_df, val_df, test_df = preprocessor.create_train_val_test_split()
    
    # Analyze class distribution
    preprocessor.analyze_class_distribution(train_df, val_df, test_df)
    
    # Create data generators
    train_gen, val_gen, test_gen = preprocessor.create_data_generators(train_df, val_df, test_df)
    
    # Visualize samples
    preprocessor.visualize_samples(train_gen)
    
    return preprocessor, train_gen, val_gen, test_gen, train_df, val_df, test_df



# Run the complete pipeline
if __name__ == "__main__":
    print("Testing data loading...")
    merged_df = test_data_loading()
    
    print("\n" + "="*50)
    print("Setting up complete data pipeline...")
    preprocessor, train_gen, val_gen, test_gen, train_df, val_df, test_df = setup_data_pipeline()
    
    print("\nData pipeline setup complete!")
    print(f"Number of classes: {len(preprocessor.class_names)}")
    print(f"Class names: {preprocessor.class_names}")