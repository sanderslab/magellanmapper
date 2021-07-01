"""Test image equality"""

from numpy import testing

from magmap.io import cli, sitk_io
from magmap.settings import config

_logger = config.logger.getChild(__name__)


class TestImgEquality:
    
    def test_reg_img(self):
        """Test equality of registered images from two base paths."""
        for key, suffix in config.reg_suffixes.items():
            if suffix is None: continue
            # load same registered image suffix for two different base paths
            # and test for Numpy array equality
            _logger.info("Loading %s for %s", key, config.filenames[:2])
            img1 = sitk_io.read_sitk_files(config.filenames[0], suffix)
            img2 = sitk_io.read_sitk_files(config.filenames[1], suffix)
            testing.assert_array_equal(img1, img2)
            _logger.info("%s are equal for %s", config.filenames[:2], key)
        

if __name__ == "__main__":
    # load CLI arguments
    cli.process_cli_args()
    test_img_equality = TestImgEquality()
    test_img_equality.test_reg_img()
