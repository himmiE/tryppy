import itertools
import json
import os
import pkgutil

from src.file_handler import FileHandler
from src.transformations.classification import Classification
from src.transformations.feature_extraction import FeatureExtraction
from src.transformations.instance_segmentation import InstanceSegmentation
from src.transformations.segmentation import Segmentation


class Tryppy:
    def __init__(self, datapath, config_filename='config.json'):
        self.incomplete_data = dict() #TODO
        config_path = datapath / config_filename

        self.ensure_config_exists(config_path)

        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
            self.file_handler = FileHandler(datapath)

    def get_features_to_save(self):
        features_to_save = []
        for feature in self.config['tasks']['feature_extraction']:
            if self.config['tasks']['feature_extraction'][feature]['save_data']['enabled']:
                features_to_save.append(feature)
                #self.file_handler.save_feature_data(feature, features[feature])
        return features_to_save

    def run(self):
        images = self.file_handler.get_input_files(self.config["input_folder_name"], keep_input_filenames=self.config["keep_input_filenames"])
        if self.config['tasks']['segmentation']['enabled']:
            segmentation_result = Segmentation().run(images)
            if self.config['tasks']['segmentation']['save_output']:
                self.file_handler.save_images_to("segmented", segmentation_result)
            images = segmentation_result

        if self.config['tasks']['instance_segmentation']['enabled']:
            instance_segmentation_result = InstanceSegmentation().run(images)
            if self.config['tasks']['instance_segmentation']['save_output']:
                self.file_handler.save_images_to('instances', instance_segmentation_result)
            images = instance_segmentation_result

        if self.config['tasks']['feature_extraction']['enabled']:
            features_to_save = self.get_features_to_save()
            features = FeatureExtraction(self.config['tasks']['feature_extraction']['grid_size']).run(images, features_to_save)
            if self.config['tasks']['feature_extraction']['save_image']['enabled']:
                self.file_handler.save_feature_images(features)
            images = features

        if self.config['tasks']['classification']['enabled']:
            classification_result = Classification().run(images)
            if self.config['tasks']['classification']['save_output']:
                self.file_handler.save_images_to('classification', classification_result)
            images = classification_result
        return images

    def ensure_config_exists(self, config_path):
        if not os.path.isfile(config_path):
            config_data = pkgutil.get_data(__name__, 'resources/default_config.json')

            # Schreibe die Datei an den Zielort
            with open(config_path, 'wb') as f:
                f.write(config_data)
            print(f"New config file has been generated at {config_path}.")
        else:
            print("Config file has been found at {config_path}.")