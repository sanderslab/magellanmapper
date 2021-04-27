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
        self.assertEqual(libmag.splitext("foo/bar/item.file.py"), (
        "foo/bar/item.file", ".py"))
        
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
        self.assertEqual(libmag.match_ext(
            "foo1/bar1/item1.file1.ext1", "foo2/bar2/item2.ext2"), (
            "foo2/bar2/item2.ext1"))
        self.assertEqual(libmag.match_ext(
            "foo1/bar1/item1.ext1", "foo2/bar2/item2.file2.ext2"), (
            "foo2/bar2/item2.file2.ext1"))
            
    def test_get_filename_without_ext(self):
        self.assertEqual(libmag.get_filename_without_ext("foo/bar/item.py"), "item")
        self.assertEqual(libmag.get_filename_without_ext("foo/bar/item"), "item")
        self.assertEqual(libmag.get_filename_without_ext(
            "foo/bar/item.file.py"), "item.file")
        self.assertEqual(libmag.get_filename_without_ext("item.py"), "item")
            
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
            
    def test_series_as_str(self):
        self.assertEqual(libmag.series_as_str("532"), "00532")
        self.assertEqual(libmag.series_as_str(""), "00000")
        self.assertEqual(libmag.series_as_str("56738"), "56738")
        self.assertEqual(libmag.series_as_str("123456789"), "123456789")
        
    def test_splice_before(self):
        self.assertEqual(libmag.splice_before("base", "file", "edit"), ("base"))
        self.assertEqual(libmag.splice_before(
        "foo/bar/item.py", "item", "file"), "foo/bar/file_item.py")
        self.assertEqual(libmag.splice_before(
        "foo/bar/item.py", "edit", "file"), "foo/bar/item_file.py")
        self.assertEqual(libmag.splice_before(
        "foo/bar/item.py", "item", "file", "/"), "foo/bar/file/item.py")
        
    def test_str_to_disp(self):
        self.assertEqual(libmag.str_to_disp("this_is_a_test"), "this is a test")
        self.assertEqual(libmag.str_to_disp(
        "       this is a test         "), "this is a test")
        self.assertEqual(libmag.str_to_disp("    this_is a_test    "), "this is a test")
        
    def test_get_int(self):
        self.assertEqual(libmag.get_int("5"), 5)
        self.assertEqual(libmag.get_int("5.6"), 5.6)
        self.assertEqual(libmag.get_int("wrong"), "wrong")
        self.assertEqual(libmag.get_int(''), '')
        
    def test_is_int(self):
        self.assertEqual(libmag.is_int(5), True)
        self.assertEqual(libmag.is_int(5.4), False)
        self.assertEqual(libmag.is_int("string"), False)
        
    def test_is_number(self):
        self.assertEqual(libmag.is_number(5), True)
        self.assertEqual(libmag.is_number(5.4), True)
        self.assertEqual(libmag.is_number("string"), False)
        
        
if __name__ == "__main__":
    unittest.main()
