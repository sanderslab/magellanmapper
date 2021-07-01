"""Unit testing for the chunking module"""

import unittest
from typing import Sequence

import numpy as np

from magmap.cv import chunking, detector
from magmap.settings import config


class TestChunking(unittest.TestCase):
    
    @staticmethod
    def stack_split_remerge(
            roi: np.ndarray, max_pixels: Sequence[int], overlap: Sequence[int]
    ) -> np.ndarray:
        """Split and remerge a stack.
        
        Args:
            roi: Region of interest.
            max_pixels: Maximum pixels along each dimension.
            overlap: Number of overlapping pixels along each dimension, to
                be added to ``max_pixels`` if possible.

        Returns:
            The remerged stack

        """
        sub_roi_slices, sub_rois_offsets = chunking.stack_splitter(
            roi.shape, max_pixels, overlap)
        print("sub_rois shape: {}".format(sub_roi_slices.shape))
        print("sub_roi_slices:\n{}".format(sub_roi_slices))
        print("overlap: {}".format(overlap))
        print("sub_rois_offsets:\n{}".format(sub_rois_offsets))
        for z in range(sub_roi_slices.shape[0]):
            for y in range(sub_roi_slices.shape[1]):
                for x in range(sub_roi_slices.shape[2]):
                    coord = (z, y, x)
                    sub_roi_slices[coord] = roi[sub_roi_slices[coord]]
                    print(coord, "shape:", sub_roi_slices[coord].shape, sub_roi_slices[coord])
        # print("sub_rois:\n{}".format(sub_roi_slices))
        merged = chunking.merge_split_stack(sub_roi_slices, max_pixels, overlap)
        print("merged:\n{}".format(merged))
        return merged
    
    def test_stack_splitter(self):
        """Test splitting and remerging."""
        roi = np.arange(5 * 4 * 4).reshape((5, 4, 4))
        print("roi:\n{}".format(roi))
        max_pixels = [1, 3, 3]
        
        merged = self.stack_split_remerge(roi, max_pixels, np.array((0, 1, 1)))
        np.testing.assert_array_equal(roi, merged)

        merged = self.stack_split_remerge(roi, max_pixels, np.array((0, 1, 2)))
        np.testing.assert_array_equal(roi, merged)

        merged = self.stack_split_remerge(roi, max_pixels, np.array((1, 1, 2)))
        np.testing.assert_array_equal(roi, merged)
        
        # test overlap generated based on resolutions
        config.resolutions = [[6.6, 1.1, 1.1]]
        merged = self.stack_split_remerge(
            roi, max_pixels, detector.calc_overlap(2))
        np.testing.assert_array_equal(roi, merged)


if __name__ == "__main__":
    unittest.main()
