#!/bin/bash
# Clrbrain unit testing
# Author: David Young, 2018
"""Unit testing for the Clrbrain package.
"""

import unittest

from clrbrain import cli
from clrbrain import config

TEST_IMG = "test.czi"

class TestImageStackProcessing(unittest.TestCase):
    
    def test_process_whole_image(self):
        config.filename = TEST_IMG
        cli.main()
        self.assertEqual(True, True)

if __name__ == "__main__":
    unittest.main()
