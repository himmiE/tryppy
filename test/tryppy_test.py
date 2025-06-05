import pathlib
import unittest
from tryppy.tryppy import Tryppy
import matplotlib

class TryptagTest(unittest.TestCase):
    def test_tryptag_run(self):
        matplotlib.use("TkAgg")
        current_dir = pathlib.Path(__file__).parent
        data_path = current_dir / "resources" / "test_data"
        tryptag = Tryppy(data_path)
        tryptag.run()
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
