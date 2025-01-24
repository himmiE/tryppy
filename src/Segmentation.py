import os
import pathlib
import numpy as np

from src.Model import Model


class Segmentation:
    def __init__(self, model_name="default", data_path=""):
        self.model_name = model_name
        self.model = self.load_model()
        self.data_path = data_path

    def set_data_path(self, data_path):
        if os.path.isdir(data_path):
            self.data_path = data_path

    def get_data_path(self):
        return self.data_path

    def load_image_data(self, custom_data_path=None):
        if not custom_data_path:
            images = np.load(self.data_path)
        else:
            images = np.load(custom_data_path)
        return images

    def change_model(self, model_name="default"):
        self.model_name = model_name
        self.model = self.load_model()

    def run(self, verbose=1):
        images = self.load_image_data()
        self.model.predict(images)

    def info(self):
        model_loaded = ""
        data_path_exists = ""
        if self.model:
            model_loaded = "not yet "
        if os.path.isdir(self.data_path):
            data_path_exists = "not "
        files = len(list(pathlib.Path(self.data_path).glob('*.img')))

        print(f"You are currently aiming to use the {self.model_name} model.")
        print(f"The model has {model_loaded}been loaded.")
        print(f"The specified path for your image data is: {self.data_path}")
        print(f"The directory could {data_path_exists}be found. It contains {files} image files")
        return

    def load_model(self):
        self.model = Model(self.model_name)
        self.model.load_model()

