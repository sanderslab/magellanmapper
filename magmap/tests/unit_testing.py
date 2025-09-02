# MagellanMapper unit testing
# Author: David Young, 2018, 2020
"""Unit testing for the MagellanMapper package.
"""

import unittest

from magmap.cv import stack_detect
from magmap.io import cli, importer, np_io
from magmap.settings import config

TEST_IMG_CZI = "test.czi"
TEST_IMG_BASE = "sample_region"
TEST_IMG_TIFF = f"{TEST_IMG_BASE}.tif"


class TestImageStackProcessing(unittest.TestCase):
    
    def test_npy00_setup(self):
        config.filename = TEST_IMG_TIFF
        config.channel = None
        cli.setup_roi_profiles(["lightsheet,4xnuc"])
    
    def test_npy01_read_tif_(self):
        import os
        print(f"Current dir: {os.getcwd()}")
        print(f"Files: {os.listdir('.')}")
        img5d = np_io.read_tif(TEST_IMG_TIFF)
        config.img5d = img5d
        assert img5d is not None
        assert img5d.img is not None
        assert img5d.meta is not None
        config.resolutions = img5d.meta[config.MetaKeys.RESOLUTIONS]
        self.assertEqual(img5d.img.shape, (1, 51, 200, 200, 2))
    
    def test_npy02_write_npy(self):
        np_io.write_npy(config.img5d.img, config.img5d.meta, TEST_IMG_TIFF)
    
    def test_npy03_read_file(self):
        import os
        print(f"Files: {os.listdir('.')}")
        img5d = importer.read_file(TEST_IMG_BASE)
        config.img5d = img5d
        assert img5d is not None
        assert img5d.img is not None
        self.assertEqual(img5d.img.shape, (1, 51, 200, 200, 2))
    
    @unittest.skip("CZI files not yet supported in this test")
    def test_import_npy_image(self):
        img5d = importer.read_file(
            config.filename, config.series)
        if img5d.img is None:
            chls, import_path = importer.setup_import_multipage(
                config.filename)
            import_md = importer.setup_import_metadata(chls, config.channel)
            img5d = importer.import_multiplane_images(
                chls, import_path, import_md, channel=config.channel)
        config.img5d = img5d
        assert img5d is not None
        assert img5d.img is not None
        self.assertEqual(img5d.img.shape, (1, 51, 200, 200, 2))
    
    def test_process_whole_image(self):
        img5d = config.img5d
        assert img5d is not None
        assert img5d.img is not None
        _, _, blobs = stack_detect.detect_blobs_blocks(
            config.filename, img5d.img, (30, 30, 8), (70, 70, 10),
            config.channel)
        self.assertEqual(len(blobs.blobs), 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
