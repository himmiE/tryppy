import pathlib
import unittest

from src.tryptag import Tryptag


class TryptagTest(unittest.TestCase):
    def test_tryptag_run(self):
        current_dir = pathlib.Path(__file__).parent
        data_path = current_dir / "resources" / "test_data"
        tryptag = Tryptag(data_path)
        tryptag.run()
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
