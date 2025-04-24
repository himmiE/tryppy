import pathlib
import unittest

from src.transformations.instance_segmentation import InstanceSegmentation


class InstanceSegmentationTest(unittest.TestCase):
    def test_get_mask_file_names(self):
        current_dir = pathlib.Path(__file__).parent
        data_path = current_dir /"resources"/"test_data"
        output_path = current_dir /"resources"/"test_data"/"output"
        instance_segmentation = InstanceSegmentation()
        filenames = instance_segmentation.get_mask_file_names(data_path)
        self.assertEqual(1, len(filenames))
        self.assertTrue(filenames[0].endswith("test_masks_TP.npy"))

    def test_run(self):
        current_dir = pathlib.Path(__file__).parent
        data_path = current_dir / "resources" / "test_data"
        output_path = current_dir / "resources" / "test_data" / "output"
        instance_segmentation = InstanceSegmentation()
        instance_segmentation.run(data_path, output_path)

        filenames = instance_segmentation.get_mask_file_names(output_path)
        self.assertEqual(1, len(filenames))



if __name__ == '__main__':
    unittest.main()
