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
        self.assertEqual(libmag.splitext("item.py"), ("item", ".py"))
        self.assertEqual(libmag.splitext("foo/bar/item"), ("foo/bar/item", ""))
        self.assertEqual(libmag.splitext("foo/bar/item.file.ext"), (
            "foo/bar/item.file", ".ext"))
        
    def test_match_ext(self):
        self.assertEqual(libmag.match_ext(
            "foo1/bar1/item1.ext1", "foo2/bar2/item2.ext2"), "foo2/bar2/item2.ext1")
        # if there is no extension in the path to match, it will retain its own extension
        self.assertEqual(libmag.match_ext(
            "foo1/bar1/item1", "foo2/bar2/item2.ext2"), "foo2/bar2/item2.ext2")
        self.assertEqual(libmag.match_ext(
            "foo1/bar1/item1.ext1", "foo2/bar2/item2"), "foo2/bar2/item2.ext1")
        self.assertEqual(libmag.match_ext(
            "foo1/bar1/item1", "foo2/bar2/item2"), "foo2/bar2/item2")
            
    def test_get_filename_without_ext(self):
        self.assertEqual(libmag.get_filename_without_ext("foo/bar/item.py"), "item")
        self.assertEqual(libmag.get_filename_without_ext("foo/bar/item"), "item")
        self.assertEqual(libmag.get_filename_without_ext(
            "foo/bar/item.file.py"), "item.file")
            
    def test_combine_paths(self):
        self.assertEqual(libmag.combine_paths(
            "foo/bar/item", "file", "_", "py"), "foo/bar/item_file.py")
        self.assertEqual(libmag.combine_paths(
            "foo/bar/item", "file.py"), "foo/bar/item_file.py")
        self.assertEqual(libmag.combine_paths(
            "foo/bar/item", "file.py", "_", "ext"), "foo/bar/item_file.ext")
        self.assertEqual(libmag.combine_paths(
            "foo/bar/item", "file.py", ext="ext"), "foo/bar/item_file.ext")
        self.assertEqual(libmag.combine_paths(
            "foo/bar/item", "file.py", "_"), "foo/bar/item_file.py")

if __name__ == "__main__":
    unittest.main()
