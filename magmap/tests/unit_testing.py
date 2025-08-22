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
        cli.setup_roi_profiles(["lightsheet,4xnuc"])
    
    def test_load_image(self):
        img5d = importer.read_file(
            config.filename, config.series)
        if img5d.img is None:
            chls, import_path = importer.setup_import_multipage(
                config.filename)
            import_md = importer.setup_import_metadata(chls, config.channel)
            img5d = importer.import_multiplane_images(
                chls, import_path, import_md, channel=config.channel)
        config.img5d = img5d
        assert(img5d is not None)
        assert(img5d.img is not None)
        self.assertEqual(img5d.img.shape, (1, 51, 200, 200, 2))
    
    def test_process_whole_image(self):
        img5d = config.img5d
        assert(img5d is not None)
        assert(img5d.img is not None)
        _, _, blobs = stack_detect.detect_blobs_blocks(
            config.filename, img5d.img, (30, 30, 8), (70, 70, 10),
            config.channel)
        self.assertEqual(len(blobs), 54)


if __name__ == "__main__":
    unittest.main(verbosity=2)
