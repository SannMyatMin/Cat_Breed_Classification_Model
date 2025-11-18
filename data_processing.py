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
        for breed_folder in os.listdir(base_dir):
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
    
    def merge_dataframe(self, image_df):
        csv_df         = pd.read_csv(self.csv_path)
        csv_df['id']   = csv_df['id'].astype(str).str.strip()
        image_df['id'] = image_df['id'].astype(str).str.strip()

        merged_df = pd.merge(image_df, csv_df[['id', 'breed']], left_on='id',
                             right_on='id', how='inner', suffixes=('_folder', '_csv'))
        breed_mismatch = merged_df[ merged_df['breed_folder'] != merged_df['breed_csv'] ]
        if len(breed_mismatch) > 0:
            print(f"Warning:: {len(breed_mismatch)} breed mismatches are found between csv and folder")
            merged_df = merged_df.drop('breed_folder', axis=1)
            merged_df = merged_df.rename(columns={'breed_csv': 'breed'})
        else:
            merged_df = merged_df.drop('breed_csv', axis=1)
            merged_df = merged_df.rename(columns={'breed_folder': 'breed'})
        
        return merged_df
    
    def _validate_image_files(self):
        valid_files   = []
        missing_files = []
        for index, row in self.df.iterrows():
            if os.path.exists(row['img_path']):
                valid_files.append(index)
            else:
                missing_files.append(index)
        self.df.loc[valid_files]
        if missing_files:
            print(f"Warning:: {len(missing_files)} files are missing")
        print(f"{len(valid_files)} files are valid")

    def load_and_validate_data(self):
        img_df  = self.create_image_dataframe()
        self.df = self.merge_dataframe(img_df)
        self._validate_image_files()
        self.class_names = sorted(self.df['breed'].unique())
        print(f"Final dataset : {len(self.df)} samples and {len(self.class_names)} breeds")
        print(f"Breeds : {self.class_names}")

        return self.df
    





