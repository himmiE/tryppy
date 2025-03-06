import os
import cv2
import numpy as np
from pathlib import Path

from PIL import Image


class FileHandler:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.input_dir = data_dir / "input"
        os.makedirs(self.input_dir, exist_ok=True)

    def _get_image_path(self, folder_name, image_name):
        return folder_name / image_name

    def _get_input_files(self):
        file_extensions = ('.jpg', '.png', '.jpeg', '.tiff', '.npy')
        return self._get_image_path(self.input_dir, file_extensions)

    def _get_image_filenames_from(self, folder_name, file_extensions='.npy'):
        filenames = []
        folder_path = self.data_dir / folder_name
        for image_name in os.listdir(folder_path):
            if image_name.endswith(file_extensions):
                filenames.append(self._get_image_path(folder_path, image_name))
        return filenames

    def load_input_image(self, image_path):
        print(f"Lade {image_path} von der Festplatte.")
        return Image.open(image_path)

    def load_npy_image(self, image_path):
        print(f"Lade {image_path} von der Festplatte.")
        return np.load(image_path)