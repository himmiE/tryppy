import json
import os
import shutil

from src.file_handler import FileHandler
from src.transformations.FeatureExtraction import FeatureExtraction
from src.transformations.InstanceSegmentation import InstanceSegmentation
from src.transformations.Segmentation import Segmentation


class Tryptag:
    def __init__(self, datapath, config_filename = 'config.json'):
        config_path = datapath / config_filename

        if not os.path.isfile(config_path):
            default_config_path = "src/resources/default_config.json"
            shutil.copyfile(default_config_path, config_path)

        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
            self.file_handler = FileHandler(datapath)

    def run(self):
        images = self.file_handler.get_input_files(self.config["input_folder_name"])
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
            features = FeatureExtraction().run(images)
            for feature in features:
                if self.config['tasks']['feature_extraction'][feature]['save_data']['enabled']:
                    self.file_handler.save_feature_data(feature, features[feature])
            if self.config['tasks']['feature_extraction']['save_image']['enabled']:
                self.file_handler.save_feature_images(features)
            images = features
        #ToDo add classification
        return images

    def run_test(self, images):
        pass
