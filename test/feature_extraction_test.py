import pathlib
import unittest
import numpy as np

from src.transformations.FeatureExtraction import FeatureExtraction


class FeatureExtractionTest(unittest.TestCase):

    feature_extraction = None

    @classmethod
    def setUpClass(cls):
        current_dir = pathlib.Path(__file__).parent
        image_dir = current_dir / "resources/test_data/masks"
        cls.feature_extraction = FeatureExtraction(image_dir)

    @classmethod
    def tearDownClass(cls):
        cls.feature_extraction = None


    def test_constructor(self):
        fe = self.feature_extraction
        self.assertEqual(len(fe.mask_images), 2)  # add assertion here

    def test_outline(self):
        fe = self.feature_extraction
        outline_data_dir = pathlib.Path(__file__).parent / "resources/test_data/outline_data"
        outline_1 = fe.get_outline_from_image(fe.mask_images[0])
        outline_2 = fe.get_outline_from_image(fe.mask_images[1])
        np.savetxt(outline_data_dir/fe.mask_images[0].split("/")[-1] , outline_1) #np.loadtxt


        # ToDo come up with ways to test this data
        # How big
        # what kind of values

    def test_get_window_from_list(self):
        fe = self.feature_extraction
        window_1 = fe.get_window_from_list(10, 0, 3)
        window_2 = fe.get_window_from_list(10, 9, 3)
        window_3 = fe.get_window_from_list(10, 3, 4)
        self.assertEqual([9, 0, 1], window_1)
        self.assertEqual([8, 9, 0], window_2)
        self.assertEqual([1, 2, 3, 4], window_3)

if __name__ == '__main__':
    unittest.main()
