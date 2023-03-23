# MagellanMapper unit testing for classification
"""Unit testing for the MagellanMapper classifier module."""

import unittest

import numpy as np

from magmap.cv import classifier
from magmap.tests import test_detector


class TestClassifier(unittest.TestCase):
    
    def test_setup_classification_roi(self):
        
        # set up blobs
        blobs = test_detector.TestDetector().make_random_blobs()
        
        # set up image
        image5d = np.zeros((1, 10, 15, 20, 2))
        
        # test ROI with complete borders
        roi, blobs_mask, blobs_shift = classifier.setup_classification_roi(
            image5d, (2, 3, 4), (5, 6, 7), blobs, 2)
        np.testing.assert_array_equal(roi.shape, (5, 8, 9, 2))
        np.testing.assert_array_equal(blobs_shift, (0, 1, 1))
        
        # test ROI exceeding the image's far edge
        roi, blobs_mask, blobs_shift = classifier.setup_classification_roi(
            image5d, (2, 3, 4), (5, 14, 7), blobs, 2)
        np.testing.assert_array_equal(roi.shape, (5, 13, 9, 2))
        np.testing.assert_array_equal(blobs_shift, (0, 1, 1))
        
        # test ROI aligned with the image's near edge
        roi, blobs_mask, blobs_shift = classifier.setup_classification_roi(
            image5d, (0, 0, 0), (5, 6, 7), blobs, 2)
        np.testing.assert_array_equal(roi.shape, (5, 7, 8, 2))
        np.testing.assert_array_equal(blobs_shift, (0, 0, 0))
        
        # test ROI whose border partially exceeds image's near edge
        roi, blobs_mask, blobs_shift = classifier.setup_classification_roi(
            image5d, (0, 1, 1), (5, 6, 7), blobs, 4)
        np.testing.assert_array_equal(roi.shape, (5, 9, 10, 2))
        np.testing.assert_array_equal(blobs_shift, (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
