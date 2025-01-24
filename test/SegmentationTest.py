import unittest

from src.Segmentation import Segmentation


class SegmentationTest(unittest.TestCase):
    def prediction_test(self):

        segmentation = Segmentation()
        segmentation.set_data_path("test/resources/test_data/test_images.npy")
        predictions = segmentation.run()

        test_masks = segmentation.load_image_data("test/resources/test_data/test_masks.npy")
        self.assertEqual(predictions, test_masks)  # add assertion here


if __name__ == '__main__':
    unittest.main()
