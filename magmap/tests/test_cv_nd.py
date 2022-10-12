# MagellanMapper unit testing for cv_nd
"""Unit testing for the MagellanMapper cv_nd module."""

import unittest

import numpy as np
from numpy import testing

from magmap.cv import cv_nd


class TestCvNd(unittest.TestCase):
    
    def test_rescale_resize(self):
        # set up test array
        img = np.zeros((2, 50, 200, 251, 3))
        
        # test rescaling array
        img_rescaled = cv_nd.rescale_resize(img[0], 0.5, multichannel=True)
        testing.assert_array_equal(img_rescaled.shape, (25, 100, 126, 3))
        
        # test resizing array
        target_size = [22, 150, 300]
        img_rescaled = cv_nd.rescale_resize(img[0], target_size, multichannel=True)
        target_size.append(3)
        testing.assert_array_equal(img_rescaled.shape, target_size)
    
    def test_angle_indices(self):
        # set up mask
        mask = np.zeros((3, 4, 5), dtype=np.uint8)
        shape = mask.shape

        # fill angled planes in z-axis direction, starting with an xy-like
        # plane positioned at z = 0, y = 0 and angled to z = 3, y = 2
        for i in range(0, shape[0]):
            mask_inds = cv_nd.angle_indices(shape, (i, 0), (shape[0], 3))
            mask[mask_inds[0], mask_inds[1], mask_inds[2]] = 1
        
        # test equality to reference
        print(mask)
        ref = np.array(
            [[[1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]],
             [[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]],
             [[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0]]]
        )
        testing.assert_array_equal(mask, ref)


if __name__ == "__main__":
    unittest.main()
