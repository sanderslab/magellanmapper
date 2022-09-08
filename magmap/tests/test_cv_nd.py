# MagellanMapper unit testing for cv_nd
"""Unit testing for the MagellanMapper cv_nd module."""

import unittest

import numpy as np
from numpy import testing

from magmap.cv import cv_nd


class TestCvNd(unittest.TestCase):
    
    def test_rescale_resize(self):
        img = np.zeros((2, 50, 200, 251, 3))
        img_rescaled = cv_nd.rescale_resize(img[0], 0.5, multichannel=True)
        testing.assert_array_equal(img_rescaled.shape, (25, 100, 126, 3))


if __name__ == "__main__":
    unittest.main()
