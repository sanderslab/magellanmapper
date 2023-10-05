# MagellanMapper unit testing for np_io
"""Unit testing for the MagellanMapper np_io module.
"""

import unittest

import numpy as np

from magmap.io import np_io


class TestLibmag(unittest.TestCase):
    
    def test_get_num_channels(self):
        # test 5D image
        img5d = np.zeros((1, 3, 4, 5, 6))
        self.assertEqual(np_io.get_num_channels(img5d), 6)

        # test 4D+channel image
        img5d_no_chl = np.zeros((1, 3, 4, 5))
        self.assertEqual(np_io.get_num_channels(img5d_no_chl), 1)

        # test 3D+channel image
        img3d = np.zeros((3, 4, 5, 6))
        self.assertEqual(np_io.get_num_channels(img3d, True), 6)

        # test 3D image
        img3d_no_chl = np.zeros((3, 4, 5))
        self.assertEqual(np_io.get_num_channels(img3d_no_chl, True), 1)


if __name__ == "__main__":
    unittest.main()
