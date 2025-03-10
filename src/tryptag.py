import json

from src.file_handler import FileHandler
from src.transformations.FeatureExtraction import FeatureExtraction
from src.transformations.InstanceSegmentation import InstanceSegmentation
from src.transformations.Segmentation import Segmentation


class Tryptag:
    def __init__(self, datapath):
        config_path = datapath / "config.json"
        #ToDo if no config_file present make default

        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
            self.file_handler = FileHandler(datapath)

    def run(self, input_folder_name="input"):
        images = self.file_handler.get_input_files(input_folder_name)
        if self.config['steps']['segmentation']['enabled']:
            segmentation_result = Segmentation().run(images)
            if self.config['steps']['segmentation']['save_output']:
                self.file_handler.save_images_to("segmented", segmentation_result)
            images = segmentation_result

        if self.config['steps']['instance_segmentation']['enabled']:
            instance_segmentation_result = InstanceSegmentation().run(images)
            if self.config['steps']['instance_segmentation']['save_output']:
                self.file_handler.save_images_to('instances', instance_segmentation_result)
            images = instance_segmentation_result

        if self.config['steps']['feature_extraction']['enabled']:
            if self.config['steps']['feature_extraction']['save_contour_image']:
                save_contour_image = True
            if self.config['steps']['feature_extraction']['save_contour_data']:
                save_contour_data = True
                #self.file_handler.save_images_to('features', features)
            features = FeatureExtraction.run(images,
                                             save_contour_data=save_contour_data,
                                             save_contour_image=save_contour_image)
            images = features
        return images

    def run_test(self, images):
        pass
