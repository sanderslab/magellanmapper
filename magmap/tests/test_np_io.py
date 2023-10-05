# MagellanMapper unit testing for np_io
"""Unit testing for the MagellanMapper np_io module.
"""

import unittest

import numpy as np

from magmap.io import np_io


class TestLibmag(unittest.TestCase):
    
    def test_get_num_channels(self):
        img5d = np.zeros((1, 3, 4, 5, 6))
        self.assertEqual(np_io.get_num_channels(img5d), 6)

        img5d_no_chl = np.zeros((1, 3, 4, 5))
        self.assertEqual(np_io.get_num_channels(img5d_no_chl), 1)

        img3d = np.zeros((3, 4, 5, 6))
        self.assertEqual(np_io.get_num_channels(img3d), 6)

        img3d_no_chl = np.zeros((3, 4, 5))
        self.assertEqual(np_io.get_num_channels(img3d_no_chl), 1)


if __name__ == "__main__":
    unittest.main()
