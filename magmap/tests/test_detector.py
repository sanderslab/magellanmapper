# MagellanMapper unit testing for detector
"""Unit testing for the MagellanMapper detector module."""

import unittest

import numpy as np

from magmap.cv import detector


class TestDetector(unittest.TestCase):
    
    def test_blob_accessors(self):
        # set up random blobs
        blobs = np.random.rand(20).reshape((5, 4))
        blobs[:, 0:3] = np.multiply(blobs[:, 0:3], 100).astype(int)
        blobs[:, 3] = blobs[:, 3] * 10
        bl = detector.Blobs(blobs)
        bl.blobs = bl.format_blobs(bl.blobs)
        print(f"Blobs:\n{bl.blobs}")

        # test blob confirmation flag
        np.testing.assert_array_equal(
            bl.get_blob_confirmed(bl.blobs), bl.blobs[:, 4])
        conf_flag = 1
        bl.set_blob_confirmed(bl.blobs, conf_flag)
        self.assertTrue(np.all(bl.get_blob_confirmed(bl.blobs) == conf_flag))
        
        # test blob truth flag
        np.testing.assert_array_equal(
            bl.get_blob_truth(bl.blobs), bl.blobs[:, 5])
        truth_flag = 2
        bl.set_blob_truth(bl.blobs, truth_flag)
        self.assertTrue(np.all(bl.get_blob_truth(bl.blobs) == truth_flag))

        # test blob channel
        np.testing.assert_array_equal(
            bl.get_blobs_channel(bl.blobs), bl.blobs[:, 6])
        channel = 3
        bl.set_blob_channel(bl.blobs, channel)
        self.assertTrue(np.all(bl.get_blobs_channel(bl.blobs) == channel))

        # test blob absolute coordinates
        np.testing.assert_array_equal(
            bl.get_blob_abs_coords(bl.blobs), bl.blobs[:, 7:10])
        abs_coords = (1, 2, 3)
        bl.set_blob_abs_coords(bl.blobs, abs_coords)
        self.assertTrue(all(np.all(
            bl.get_blob_abs_coords(bl.blobs) == abs_coords, axis=1)))


if __name__ == "__main__":
    unittest.main()
