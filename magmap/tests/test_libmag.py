# MagellanMapper unit testing for libmag
"""Unit testing for the MagellanMapper libmag module.
"""

import unittest


from magmap.io import libmag

class TestLibmag(unittest.TestCase):
    
    
    def test_insert_before_ext(self):
        self.assertEqual(libmag.insert_before_ext(
            "foo/bar/item.py", "totest", "_"), "foo/bar/item_totest.py")
        
    

if __name__ == "__main__":
    unittest.main()
