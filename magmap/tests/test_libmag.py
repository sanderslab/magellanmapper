# MagellanMapper unit testing for libmag
"""Unit testing for the MagellanMapper libmag module.
"""

import unittest


from magmap.io import libmag

class TestLibmag(unittest.TestCase):
    
    
    def test_insert_before_ext(self):
        self.assertEqual(libmag.insert_before_ext(
            "foo/bar/item.py", "totest", "_"), "foo/bar/item_totest.py")
        self.assertEqual(libmag.insert_before_ext(
            "foo/bar/item.py", "totest"), "foo/bar/itemtotest.py")
        self.assertEqual(libmag.insert_before_ext(
            "foo/bar/item", "totest", "_"), "foo/bar/item_totest")
            
    def test_splitext(self):
        self.assertEqual(libmag.splitext("foo/bar/item.py"), ("foo/bar/item", ".py"))
        
    def test_match_ext(self):
        self.assertEqual(libmag.match_ext(
            "foo1/bar1/item1.ext1", "foo2/bar2/item2.ext2"), "foo2/bar2/item2.ext1")
            
    def test_get_filename_without_ext(self):
        self.assertEqual(libmag.get_filename_without_ext("foo/bar/item.py"), "item")

if __name__ == "__main__":
    unittest.main()
