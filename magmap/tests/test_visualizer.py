# MagellanMapper unit testing for visualizer
"""Unit testing for the MagellanMapper visualizer module."""

import unittest

from magmap.gui import roi_editor, visualizer


class TestVisualizer(unittest.TestCase):
    
    def test_validate_pref(self):
        # select from enum
        default = roi_editor.ROIEditor.CircleStyles.CIRCLES.value
        circles = roi_editor.ROIEditor.CircleStyles.REPEAT_CIRCLES.value
        pref = visualizer.Visualization.validate_pref(
            circles, default, roi_editor.ROIEditor.CircleStyles)
        self.assertEqual(pref, circles)

        circles += "!"
        pref = visualizer.Visualization.validate_pref(
            circles, default, roi_editor.ROIEditor.CircleStyles)
        self.assertEqual(pref, default)
        
        # select from list
        default = visualizer.Visualization._DEFAULTS_PLANES_2D[0]
        plane = visualizer.Visualization._DEFAULTS_PLANES_2D[1]
        pref = visualizer.Visualization.validate_pref(
            plane, default, visualizer.Visualization._DEFAULTS_PLANES_2D)
        self.assertEqual(pref, plane)

        plane = None
        pref = visualizer.Visualization.validate_pref(
            plane, default, visualizer.Visualization._DEFAULTS_PLANES_2D)
        self.assertEqual(pref, default)
    
    def test_get_changed_options(self):
        curr_options = [visualizer.RegionOptions.BOTH_SIDES.value]
        prev_options = {e: False for e in visualizer.RegionOptions}
        options, changed = visualizer.Visualization.get_changed_options(
            curr_options, prev_options, visualizer.RegionOptions)
        self.assertTrue(options[visualizer.RegionOptions.BOTH_SIDES])
        self.assertFalse(options[visualizer.RegionOptions.APPEND])
        self.assertFalse(changed[visualizer.RegionOptions.APPEND])
        self.assertTrue(changed[visualizer.RegionOptions.BOTH_SIDES])
        
        curr_options = [visualizer.RegionOptions.APPEND.value]
        options, changed = visualizer.Visualization.get_changed_options(
            curr_options, options, visualizer.RegionOptions)
        self.assertFalse(options[visualizer.RegionOptions.BOTH_SIDES])
        self.assertTrue(options[visualizer.RegionOptions.APPEND])
        self.assertTrue(changed[visualizer.RegionOptions.APPEND])
        self.assertTrue(changed[visualizer.RegionOptions.BOTH_SIDES])
        self.assertFalse(changed[visualizer.RegionOptions.INCL_CHILDREN])


if __name__ == "__main__":
    unittest.main()
