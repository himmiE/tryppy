import os
import numpy as np
from pathlib import Path
from PIL import Image


class FileHandler:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        #self.input_dir = data_dir / "input"
        #os.makedirs(self.input_dir, exist_ok=True)

    def get_image_path(self, folder_name, image_name):
        return folder_name / image_name

    def get_input_files(self, input_folder_name="input"):
        file_extensions = ('.jpg', '.png', '.jpeg', '.tiff', '.npy')
        input_image_filenames = self.get_image_filenames_from(input_folder_name, file_extensions=file_extensions)
        input_images = {i: np.array(Image.open(image)) if not str(image).endswith(".npy") else np.load(image)
                        for i, image in enumerate(input_image_filenames)}
        return input_images

    def get_image_filenames_from(self, folder_name, file_extensions=None):
        if file_extensions is None:
            file_extensions = "npy"
        filenames = []
        folder_path = self.data_dir / folder_name
        for image_name in os.listdir(folder_path):
            if image_name.endswith(file_extensions):
                filenames.append(self.get_image_path(folder_path, image_name))

        count_files = len(filenames)
        print(f"The path for your input data is: {self.data_dir / folder_name}")
        print(f"The directory contains {count_files} suitable image files.")
        return filenames

    def save_image(self, image_path, image):
        np.save(image_path, image)

    def save_images_to(self, folder_name, images):
        for image in images:
            image_path = self.data_dir / folder_name / f"{image}.npy"
            np.save(image_path, images[image])

