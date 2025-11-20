import os
import pandas as pd
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
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
    
    def create_train_validation_test_split(self):
        train_val_df, test_df = train_test_split(
            self.df, test_size=self.test_split, stratify=self.df['breed'], random_state=42 )
        train_df, validation_df = train_test_split(
            train_val_df, test_size=self.validation_split, stratify=train_val_df['breed'], random_state=42 )
        
        return train_df, validation_df, test_df
    
    def create_data_generator(self, train_df, val_df, test_df):
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2, 
            height_shift_range=0.2,
            horizontal_flip=True, 
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            shear_range=0.15,
            fill_mode="nearest" )
         
        
        val_test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col="img_path",
            y_col="breed",
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True )
        
        validation_generator = val_test_datagen.flow_from_dataframe(
            val_df,
            x_col="img_path",
            y_col="breed",
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False )
        
        test_generator = val_test_datagen.flow_from_dataframe(
            test_df,
            x_col="img_path",
            y_col="breed",
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False )
        
        self.class_names = list(train_generator.class_indices.keys())
        self.class_indices = train_generator.class_indices

        return train_generator, validation_generator, test_generator
    
    def analyze_class_distribution(self, train_df, val_df, test_df):
        fig,ax = plt.subplots(4, 1, figsize=(24, 47))

        train_df_count = train_df['breed'].value_counts()
        ax[0].bar(range(len(train_df_count)), train_df_count.values, width=0.8)
        ax[0].set_title("Training Set", fontweight="bold", fontsize=24)
        ax[0].title.set_position([0.5, 1.2])
        ax[0].set_ylabel("Count")
        ax[0].set_xticks(range(len(train_df_count)))
        ax[0].set_xticklabels(train_df_count.index, rotation=90, fontsize=15)

        val_df_count = val_df['breed'].value_counts()
        ax[1].bar(range(len(val_df_count)), val_df_count.values)
        ax[1].set_title("Validation Set", fontweight="bold", fontsize=24)
        ax[1].title.set_position([0.5, 1.2])
        ax[1].set_ylabel("Count")
        ax[1].set_xticks(range(len(val_df_count)))
        ax[1].set_xticklabels(val_df_count.index, rotation=90, fontsize=15)

        test_df_count = test_df['breed'].value_counts()
        ax[2].bar(range(len(test_df_count)), test_df_count.values)
        ax[2].set_title("Testing Set", fontweight="bold", fontsize=24)
        ax[2].title.set_position([0.5, 1.2])
        ax[2].set_ylabel("Count")
        ax[2].set_xticks(range(len(test_df_count)))
        ax[2].set_xticklabels(test_df_count.index, rotation=90, fontsize=15)

        ax[3].text(0, 0.8, "Class Distribution Status", fontsize=24, fontweight="bold")
        ax[3].text(0, 0.7, f"Maximum Samples per Class = {max(train_df_count)}", fontsize=15)
        ax[3].text(0, 0.6, f"Minimum Samples per Class = {min(train_df_count)}", fontsize=15)
        ax[3].text(0, 0.5, f"Average Samples per Class = {train_df_count.mean():.2f}", fontsize=15)
        ax[3].set_xlim(0,1)
        ax[3].set_ylim(0,1)
        ax[3].axis('off')

        plt.tight_layout()
        plt.show()

def set_up_data_pipeline():
    data_processor = CatDataProcessor(dataset_dir="dataset", img_size=(380, 380))
    train_df, validation_df, test_df = data_processor.create_train_validation_test_split()
    train, validation, test = data_processor.create_data_generator(train_df, validation_df, test_df)
    
    return data_processor, train, validation, test, train_df, validation_df, test_df



