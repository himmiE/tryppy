import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tifffile import tifffile

from src.feature_visualizer import FeatureVisualizer


class FileHandler:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.feature_visualizer = FeatureVisualizer()

    def get_image_path(self, folder_name, image_name):
        return folder_name / image_name

    def get_input_files(self, input_folder_name="input", keep_input_filenames=True):
        file_extensions = ('.jpg', '.png', '.jpeg', '.tif', '.npy')
        input_image_filenames = self.get_image_filenames_from(input_folder_name, file_extensions=file_extensions)
        input_images = {}
        for i, image_filename in enumerate(input_image_filenames):
            if keep_input_filenames:
                image_key = os.path.basename(image_filename).split(".")[0]
            else:
                image_key = i

            if str(image_filename).endswith(".npy"):
                input_images[image_key] = np.load(image_filename)
            #elif str(image).endswith(".tiff"):
            # TODO make work with Tiff-files
            elif str(image_filename).endswith(".tif"):
                tiff_data = tifffile.imread(image_filename)
                image_data = self.get_mask_from_tiff(tiff_data)
                input_images[image_key] = image_data
        keys_to_extract = ['test_images_TP_0', 'test_images_TP_1', 'test_images_TP_2']
        input_images = {k: input_images[k] for k in keys_to_extract if k in input_images}
        return input_images

    def get_mask_from_tiff(self, image):
        image_data = np.array(image)[0]
        phase_channel = [image[:, :, 0] for image in image_data]
        # dna_channel = [image[:, :, 2] for image in image_data]
        # ToDo make tiff files work as input
        #image = tf.keras.layers.Resizing(height=320, width=320)(image)
        #image = image / np.max(image)
        #image = tf.cast(image, dtype=tf.float64)
        return phase_channel

    def get_image_filenames_from(self, folder_name, file_extensions=None):
        if file_extensions is None:
            file_extensions = "npy"
        filenames = []
        folder_path = self.data_dir / folder_name
        for image_name in os.listdir(folder_path):
            if image_name.endswith(file_extensions):
                filenames.append(self.get_image_path(folder_path, image_name))

        count_files = len(filenames)
        print(f"The path for your data is: {folder_path}")
        print(f"The directory contains {count_files} suitable image files.")
        return filenames

    def save_images_to(self, folder_name, images):
        folder_path = self.data_dir / folder_name
        os.makedirs(folder_path, exist_ok=True)
        for name, image in images.items():
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

