import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tifffile import tifffile
import skimage

from src.feature_visualizer import FeatureVisualizer


class FileHandler:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.feature_visualizer = FeatureVisualizer()

    def get_image_path(self, folder_name, image_name):
        return folder_name / image_name

    def load_file(self, filepath):
        if str(filepath).endswith(".npy"):
            return np.load(filepath)
        # elif str(image).endswith(".tiff"):
        # TODO make work with Tiff-files
        elif str(filepath).endswith(".tif"):
            tiff_data = skimage.io.imread(filepath, plugin="pil")
            return tiff_data

    def save_df(self, df, folder="dataframe"):
        folder_path = self.data_dir / folder
        os.makedirs(folder_path, exist_ok=True)
        file_path = folder_path / 'tryppy_features.csv'
        df.to_csv(file_path, index=False)

    def get_input_files(self, input_folder_name="input", extension='tif'):
        #keys_to_extract = ['Tb927.3.930_4_N_1_118_0']

        file_extensions = ('.jpg', '.png', '.jpeg', '.tif', '.npy')
        input_image_filenames = self.get_image_filenames_from(input_folder_name, file_extensions=file_extensions)
        input_images = {}
        for folder in input_image_filenames:
            if isinstance(input_image_filenames[folder], dict):
                for sub_dir, file_names in input_image_filenames[folder].items():
                    input_images[sub_dir] = {}
                    image_keys = ['.'.join(os.path.basename(f).split(".")[:-1]) for f in file_names]
                    temp_file_dict = {image_key: self.load_file(Path(f"{folder}/{sub_dir}/{image_key}{extension}")) for image_key in image_keys}
                    #temp_file_dict = {k: v for k, v in temp_file_dict.items() if k in keys_to_extract}
                    input_images[sub_dir].update(temp_file_dict)
            else:
                file_names = input_image_filenames[folder]
                image_keys = ['.'.join(os.path.basename(f).split(".")[:-1]) for f in file_names]
                temp_file_dict = {image_key: self.load_file(Path(f"{folder}/{image_key}{extension}")) for
                                  image_key in image_keys}
                input_images.update(temp_file_dict)


        return input_images

    '''def get_mask_from_tiff(self, image):
        #image_data = np.array(image)[0]
        phase_channel = np.array(image)[0]
        # dna_channel = np.array(image)[1] for image in image_data]
        # ToDo make tiff files work as input
        #image = tf.keras.layers.Resizing(height=320, width=320)(image)
        #image = image / np.max(image)
        #image = tf.cast(image, dtype=tf.float64)
        return phase_channel'''

    def get_image_filenames_from(self, folder_name, file_extensions=None):
        if file_extensions is None:
            file_extensions = "npy"
        folder_path = self.data_dir / folder_name
        filenames = {folder_path: []}
        for image_name in os.listdir(folder_path):
            if image_name.endswith(file_extensions):
                filenames[folder_path].append(image_name)

        if not filenames[folder_path]:
            filenames = {folder_path: {}}
            dirs = [p.relative_to(folder_path) for p in folder_path.iterdir() if p.is_dir()]
            for image_dir in dirs:
                filenames[folder_path][image_dir] = []
                for image_name in os.listdir(folder_path/image_dir):
                    if image_name.endswith(file_extensions):
                        filenames[folder_path][image_dir].append(image_name)

        count_files = len(filenames)
        print(f"The path for your data is: {folder_path}")
        print(f"The directory contains {count_files} suitable image files.")
        return filenames

    def save_images_to(self, folder_name, images):
        folder_path = self.data_dir / folder_name
        os.makedirs(folder_path, exist_ok=True)
        for name, image in images.items():
            if isinstance(image, dict):
                new_folder_name = f"{folder_name}/{name}"
                self.save_images_to(new_folder_name, image)
            else:
                image_path = self.data_dir / folder_name / f"{name}.npy"
                np.save(file=image_path, arr=image)

    def save_as_json_files(self, folder_name, filename, data):
        folder_path = self.data_dir / folder_name
        os.makedirs(folder_path, exist_ok=True)
        file_path = self.data_dir / folder_name / f"{filename}.json"
        file = open(file_path, "w")
        json.dump(data, file)

    def save_numpy_data(self, folder_name, filename, data_dict):
        folder_path = self.data_dir / "raw_data_structures" / folder_name
        os.makedirs(folder_path, exist_ok=True)
        for file_name in data_dict:
            file_path = folder_path / filename
            np.save(file_path, data_dict[file_name])

    def save_feature_data(self, feature, param):
        if feature == "contour":
            pass
        pass

    def save_plot(self, folder_name, filename, plt):
        folder_path = self.data_dir / folder_name
        os.makedirs(folder_path, exist_ok=True)
        file_path = self.data_dir / folder_name / f"{filename}.png"
        plt.savefig(file_path)
        pass

