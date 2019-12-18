#!/bin/bash
# Clrbrain unit testing
# Author: David Young, 2018, 2019
"""Unit testing for the Clrbrain package.
"""

import unittest

from magmap import cli
from magmap import config
from magmap import importer
from magmap import profiles

TEST_IMG = "test.czi"


class TestImageStackProcessing(unittest.TestCase):
    
    def setUp(self):
        config.filename = TEST_IMG
        config.series = 0
        config.channel = None
        profiles.update_process_settings(
            config.process_settings, "lightsheet_v02.2")
        cli.proc_type = cli.PROC_TYPES[2]
    
    def test_load_image(self):
        image5d = importer.read_file(
            config.filename, config.series, channel=config.channel, load=False)
        self.assertEqual(image5d.shape, (1, 51, 200, 200, 2))
    
    def test_process_whole_image(self):
        _, _, blobs = cli.process_file(
            config.filename, config.series, (30, 30, 8), (70, 70, 10))
        self.assertEqual(len(blobs), 195)


if __name__ == "__main__":
    unittest.main(verbosity=2)
