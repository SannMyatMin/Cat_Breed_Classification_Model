import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from sklearn.model_selection import train_test_split

class CatDataProcessor:
    def __init__(self, dataset_dir, csv_name="cat_breed.csv", images_dir="images", img_size=(224,224),
                 batch_size=32, validation_split=0.2, test_split=0.1):
        self.dataset_dir      = Path(dataset_dir)
        self.csv_path         = self.dataset_dir / csv_name
        self.images_dir       = self.dataset_dir / images_dir
        self.image_size       = img_size
        self.batch_size       = batch_size
        self.validation_split = validation_split
        self.test_split       = test_split
        self.df               = None
        self.class_names      = []

    def create_image_dataframe(self):
        image_data = []
        base_dir = Path("dataset/images")
        for breed_folder in os.path.lisdir(base_dir):
            folder_path = os.path.join(base_dir, breed_folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(('.jpg', ".png", ".jpeg")):
                        filename_without_extension = os.path.splitext(filename)[0]
                        img_id = filename_without_extension.split("_")[0]
                        image_data.append({
                            "id": img_id,
                            "img_path": os.path.join(folder_path, filename),
                            "breed": breed_folder
                        })
        image_dataframe = pd.DataFrame(image_data)
        return image_dataframe
