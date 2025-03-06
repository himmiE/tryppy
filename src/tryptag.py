from src.transformations.FeatureExtraction import FeatureExtraction
from src.transformations.Segmentation import Segmentation


class Tryptag:
    def __init__(self):
        self.segmentation = Segmentation()
        self.feature_extraction = FeatureExtraction()
