import os
import pathlib
import unittest

from numpy import mean

from src.Segmentation import Segmentation


class SegmentationTest(unittest.TestCase):
    def test_prediction(self):

        segmentation = Segmentation()
        current_dir = pathlib.Path(__file__).parent
        testdata_path_x = current_dir / "resources" / "images" / "test_images_TP.npy"
        testdata_path_y = current_dir / "resources" / "images" / "test_masks_TP.npy"
        segmentation.set_data_path(testdata_path_x)
        predictions = segmentation.run()

        test_masks = segmentation.load_image_data(testdata_path_y)
        print(predictions.shape)
        print(test_masks.shape)
        predictions = predictions.reshape(201, 320, 320)
        difference = mean(test_masks - predictions)
        self.assertGreater(0.1, difference)  # add assertion here


if __name__ == '__main__':
    unittest.main()
