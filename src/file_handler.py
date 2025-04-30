import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

from src.feature_visualizer import FeatureVisualizer


class FileHandler:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.feature_visualizer = FeatureVisualizer()

    def get_image_path(self, folder_name, image_name):
        return folder_name / image_name

    def get_input_files(self, input_folder_name="input", keep_input_filenames=True):
        file_extensions = ('.jpg', '.png', '.jpeg', '.tiff', '.npy')
        input_image_filenames = self.get_image_filenames_from(input_folder_name, file_extensions=file_extensions)
        input_images = {}
        for i, image_filename in enumerate(input_image_filenames):
            if str(image_filename).endswith(".npy"):
                if keep_input_filenames:
                    image_key = os.path.basename(image_filename).split(".")[0]
                else:
                    image_key = i
                input_images[image_key] = np.load(image_filename)
            #elif str(image).endswith(".tiff"):
            # TODO make work with Tiff-files
            '''else:
                image_file = Image.open(image)
                image_data = np.array(image_file)
                image_file.close()
                image_data = self.pre_process_tiff_file(image_data)
                input_images[image_key] = image_data'''
        return input_images

    def pre_process_tiff_file(self, image):
        # phase_channel = [image[:, :, 0] for image in image_data]
        # dna_channel = [image[:, :, 2] for image in image_data]
        # ToDo make tiff files work as input
        image = tf.keras.layers.Resizing(height=320, width=320)(image)
        image = image / np.max(image)
        image = tf.cast(image, dtype=tf.float64)
        return image

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

    def save_numpy_data(self, folder_name, filename, data):
        folder_path = self.data_dir / folder_name
        os.makedirs(folder_path, exist_ok=True)
        file_path = self.data_dir / folder_name / filename
        np.save(file_path, data)

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

