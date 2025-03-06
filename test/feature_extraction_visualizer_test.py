import pathlib
import unittest

from src.feature_extraction_visualizer import FeatureExtractionVisualizer


class FeatureExtractionVisualizerTest(unittest.TestCase):

    visualizer = None

    @classmethod
    def setUpClass(cls):
        current_dir = pathlib.Path(__file__).parent
        image_dir = current_dir / "resources/test_data/masks"
        output_dir = current_dir / "resources/test_data/output"
        cls.feature_extraction = FeatureExtractionVisualizer(image_dir, output_dir)

    @classmethod
    def tearDownClass(cls):
        cls.feature_extraction = None

    def test_plot_outline(self):
        visualizer = self.visualizer
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
