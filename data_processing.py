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

    
