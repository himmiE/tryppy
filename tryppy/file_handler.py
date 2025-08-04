import json
import os
import pathlib
import sys
import urllib

import numpy as np
from pathlib import Path
import skimage
import joblib
import pkgutil
import io
import sklearn

class FileHandler:
    def __init__(self, data_dir, config):
        self.data_dir = Path(data_dir)
        self.config = config

    def get_image_path(self, folder_name, image_name):
        return folder_name / image_name

    def load_file(self, filepath):
        if str(filepath).endswith(".npy"):
            return np.load(filepath)
        elif str(filepath).endswith(".tif"):
            tiff_data = skimage.io.imread(filepath, plugin="pil")
            return tiff_data

    def save_df(self, df, folder="dataframe"):
        folder_path = self.data_dir / folder
        os.makedirs(folder_path, exist_ok=True)
        file_path = folder_path / 'tryppy_features.csv'
        df.to_csv(file_path, index=False)

    def get_input_files(self, input_folder_name="input", extension='tif'):
        file_extensions = ('.jpg', '.png', '.jpeg', '.tif', '.npy')
        input_image_filenames = self.get_image_filenames_from(input_folder_name, file_extensions=file_extensions)
        input_images = {}
        for folder in input_image_filenames:
            if isinstance(input_image_filenames[folder], dict):
                for sub_dir, file_names in input_image_filenames[folder].items():
                    input_images[sub_dir] = {}
                    image_keys = ['.'.join(os.path.basename(f).split(".")[:-1]) for f in file_names]
                    temp_file_dict = {image_key: self.load_file(Path(f"{folder}/{sub_dir}/{image_key}{extension}")) for image_key in image_keys}
                    input_images[sub_dir].update(temp_file_dict)
            else:
                file_names = input_image_filenames[folder]
                image_keys = ['.'.join(os.path.basename(f).split(".")[:-1]) for f in file_names]
                temp_file_dict = {image_key: self.load_file(Path(f"{folder}/{image_key}{extension}")) for
                                  image_key in image_keys}
                input_images.update(temp_file_dict)
        return input_images

    def count_leaf_values(self, d):
        count = 0
        for v in d.values():
            if isinstance(v, dict):
                count += self.count_leaf_values(v)
            else:
                count += 1
        return count

    def get_image_filenames_from(self, folder_name, file_extensions=None):
        if file_extensions is None:
            file_extensions = "npy"
        folder_path = self.data_dir / folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"❌ The required folder '{folder_path}' was not found.")
            print("The folder has been created now. You need to include the associated data into the directory"
                  "and restart the program")
            sys.exit(1)
        filenames = {folder_path: []}
        for image_name in os.listdir(folder_path):
            if image_name.endswith(file_extensions):
                filenames[folder_path].append(image_name)
        count_files = len(filenames)

        if not filenames[folder_path]:
            filenames = {folder_path: {}}
            dirs = [p.relative_to(folder_path) for p in folder_path.iterdir() if p.is_dir()]
            for image_dir in dirs:
                filenames[folder_path][image_dir] = []
                for image_name in os.listdir(folder_path/image_dir):
                    if image_name.endswith(file_extensions):
                        filenames[folder_path][image_dir].append(image_name)
        count_files = self.count_leaf_values(filenames)

        if not filenames[folder_path]:
            print(f"❌ No {file_extensions}-files have been found at '{folder_path}'.")
            print("Please include the associated data into the directory and restart the program.")
            sys.exit(1)


        print(f"The path for your data is: {folder_path}")
        print(f"The directory contains {count_files} suitable image files.")
        print(filenames)
        return filenames

    def save_images_to(self, folder_name, images):
        if folder_name:
            folder_path = self.data_dir / folder_name
        else:
            folder_path = self.data_dir
        os.makedirs(folder_path, exist_ok=True)
        for name, image in images.items():
            if isinstance(image, dict):
                new_folder_name = f"{folder_name}/{name}"
                self.save_images_to(new_folder_name, image)
            else:
                image_path = self.data_dir / folder_name / f"{name}.npy"
                np.save(file=image_path, arr=image)

    def save_as_json_files(self, folder_name, filename, data):
        if folder_name:
            folder_path = self.data_dir / folder_name
        else:
            folder_path = self.data_dir
        os.makedirs(folder_path, exist_ok=True)
        file_path = folder_path / f"{filename}.json"
        file = open(file_path, "w")
        json.dump(data, file)

    def save_numpy_data(self, folder_name, filename, data_dict):
        folder_path = self.data_dir / "raw_data_structures" / folder_name
        os.makedirs(folder_path, exist_ok=True)
        for file_name in data_dict:
            file_path = folder_path / filename
            np.save(file_path, data_dict[file_name])

    def save_plot(self, folder_name, filename, plt):
        folder_path = self.data_dir / folder_name
        os.makedirs(folder_path, exist_ok=True)
        file_path = self.data_dir / folder_name / f"{filename}.png"
        plt.savefig(file_path)
        pass

    def load_rf(self):
        model_data = pkgutil.get_data(__name__, 'resources/rf_model/random_forest_model.joblib')
        label_encoder_data = pkgutil.get_data(__name__, 'resources/rf_model/label_encoder.joblib')

        # Bytes in ein File-like Object umwandeln
        model = joblib.load(io.BytesIO(model_data))
        label_encoder = joblib.load(io.BytesIO(label_encoder_data))
        return model, label_encoder

    def ensure_unet_model(self):
        current_dir = pathlib.Path(__file__).parent
        local_path = current_dir / self.config["weights_path"]
        unet_found = os.path.exists(local_path)
        if unet_found:
            print("found weights for unet")
            return
        else:
            print(f"No weights found at {local_path}. Downloading unet model weights...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(self.config["model_url"], local_path)
