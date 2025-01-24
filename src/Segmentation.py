import os
import pathlib


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

    def change_model(self, model_name="default"):
        self.model_name = model_name
        self.model = self.load_model()

    def run(self, verbose=1):
        self.model.generate_masks()

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
        filename = os.getcwd() + "/" + self.model_name

    def  load_data(self, datasource="default"):
