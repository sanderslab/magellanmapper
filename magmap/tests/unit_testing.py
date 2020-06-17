# MagellanMapper unit testing
# Author: David Young, 2018, 2020
"""Unit testing for the MagellanMapper package.
"""

import unittest

from magmap.cv import stack_detect
from magmap.io import cli
from magmap.io import importer
from magmap.settings import config

TEST_IMG = "test.czi"


class TestImageStackProcessing(unittest.TestCase):
    
    def setUp(self):
        config.filename = TEST_IMG
        config.channel = None
        cli.setup_profiles(["lightsheet,4xnuc"], None, None)
    
    def test_load_image(self):
        config.image5d = importer.read_file(
            config.filename, config.series)
        if config.image5d is None:
            chls, import_path = importer.setup_import_bioformats(
                config.filename)
            config.image5d = importer.import_bioformats(
                chls, import_path, channel=config.channel)
        self.assertEqual(config.image5d.shape, (1, 51, 200, 200, 2))
    
    def test_process_whole_image(self):
        _, _, blobs = stack_detect.detect_blobs_large_image(
            config.filename, config.image5d, (30, 30, 8), (70, 70, 10))
        self.assertEqual(len(blobs), 54)


if __name__ == "__main__":
    unittest.main(verbosity=2)
