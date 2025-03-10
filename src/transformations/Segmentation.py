import os
import pathlib
import numpy as np
from PIL import Image

from src.transformations.Model import Model


class Segmentation:
    def __init__(self, model_name="default"):
        self.model_name = model_name
        self.model = None
        self.load_model()

    def change_model(self, model_name="default"):
        self.model_name = model_name
        self.load_model()

    def run(self, images, verbose=1):
        result = dict()
        for nr, image in images.items():
            segmentation = self.model.predict(image)
            temp_result = {i: v for i, v in enumerate(segmentation)}
            result.update(temp_result)
        return result

    def info(self, images):
        model_loaded = ""
        if self.model:
            model_loaded = "not yet "
        #files = len(list(pathlib.Path(self.data_path).glob('*.img')))

        print(f"You are currently aiming to use the {self.model_name} model.")
        print(f"The model has {model_loaded}been loaded.")
        #print(f"The specified path for your image data is: {self.data_path}")
        #print(f"The directory could {data_path_exists}be found. It contains {files} image files")
        return

    def load_model(self):
        self.model = Model(self.model_name)
        self.model.load_model()





