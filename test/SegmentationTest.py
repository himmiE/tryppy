import pathlib
import unittest

import numpy as np
from numpy import mean

from src.file_handler import FileHandler
from src.transformations.Segmentation import Segmentation


class SegmentationTest(unittest.TestCase):
    def test_prediction(self):

        segmentation = Segmentation()
        current_dir = pathlib.Path(__file__).parent
        testdata_path_x = current_dir / "resources" / "test_data" / "test_images_TP.npy"
        testdata_path_y = current_dir / "resources" / "test_data" / "test_masks_TP.npy"

        image_raw = np.load(testdata_path_x)
        image_eval = np.load(testdata_path_y)
        predictions = segmentation.run({0:image_raw})

        predictions = np.vstack(list(predictions.values()))
        predictions = predictions.reshape(201, 320, 320)

        difference = mean(image_eval - predictions)
        self.assertGreater(0.1, difference)  # 0.1 greater than difference of prediction to expected


if __name__ == '__main__':
    unittest.main()
