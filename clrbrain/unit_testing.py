#!/bin/bash
# Clrbrain unit testing
# Author: David Young, 2018
"""Unit testing for the Clrbrain package.
"""

import unittest

from clrbrain import cli
from clrbrain import config
from clrbrain import importer

TEST_IMG = "test.czi"

class TestImageStackProcessing(unittest.TestCase):
    
    def setUp(self):
        config.filename = TEST_IMG
        config.series = 0
        config.channel = None
    
    def test_load_image(self):
        image5d = importer.read_file(
            config.filename, config.series, channel=config.channel, load=False)
        self.assertEqual(image5d.shape, (1, 51, 200, 200, 2))
    
    def test_process_whole_image(self):
        # TODO: simply checking for exceptions now; need to get segments
        cli.main()
        self.assertEqual(True, True)

if __name__ == "__main__":
    unittest.main(verbosity=2)
