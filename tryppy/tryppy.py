import json
import os
import pkgutil

from tryppy.file_handler import FileHandler
from tryppy.transformations.classification import Classification
from tryppy.transformations.feature_extraction import FeatureExtraction
from tryppy.transformations.instance_segmentation import InstanceSegmentation
from tryppy.transformations.segmentation import MaskExtraction

class Tryppy:
    def __init__(self, datapath, config_filename='config.json'):
        config_path = datapath / config_filename

        os.makedirs(datapath, exist_ok=True)
        self.ensure_config_exists(config_path)

        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            self.config = config
            self.file_handler = FileHandler(datapath, config)

        import warnings
        warnings.filterwarnings("ignore")

    def get_features_to_save(self):
        features_to_save = []
        for feature in self.config['tasks']['feature_extraction']['save_data']:
            features_to_save.append(feature)
        return features_to_save


    def run(self):
        images = self.file_handler.get_input_files(self.config["input_folder_name"], self.config["input_extension"])
        full_image_masks=[]
        masks=[]
        result = {}
        if self.config['tasks']['full_image_masks']['enabled']:
            print("generating mask from input")
            self.file_handler.ensure_unet_model()
            full_image_masks = MaskExtraction(self.config['weights_path']).run(images)
            if self.config['tasks']['full_image_masks']['save_output']:
                self.file_handler.save_images_to("full_image_masks", full_image_masks)

        if self.config['tasks']['crop']['enabled']:
            print("cropping masks and images")
            crops, masks = InstanceSegmentation(self.file_handler).run(images, full_image_masks)
            if self.config['tasks']['crop']['save_crop']:
                self.file_handler.save_images_to("crops", crops)
            if self.config['tasks']['crop']['save_mask']:
                self.file_handler.save_images_to("mask", masks)
            result = crops

        if self.config['tasks']['feature_extraction']['enabled']:
            print("starting feature extraction")
            features_to_save = self.get_features_to_save()
            feature_df = FeatureExtraction(self.config, self.file_handler).run(result, masks, features_to_save)
            result = feature_df

        if self.config['tasks']['classification']['enabled']:
            classification_result, count_classes = Classification(self.file_handler).run(result)
            if self.config['tasks']['classification']['save_output']:
                self.file_handler.save_as_json_files(None,'classification', classification_result)
                self.file_handler.save_as_json_files(None, 'class_count', count_classes)
            result = classification_result
        return result

    def ensure_config_exists(self, config_path):
        if not os.path.isfile(config_path):
            config_data = pkgutil.get_data("tryppy", "resources/default_config.json")

            # Schreibe die Datei an den Zielort
            with open(config_path, 'wb') as f:
                f.write(config_data)
            print(f"New config file has been generated at {config_path}.")
        else:
            print("Config file has been found at {config_path}.")