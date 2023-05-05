# MagellanMapper unit testing for libmag
"""Unit testing for the MagellanMapper libmag module.
"""

import unittest


from magmap.io import libmag


class TestLibmag(unittest.TestCase):
    
    def test_pad_seq(self):
        seq = ["a", "b", "c"]
        self.assertSequenceEqual(libmag.pad_seq(list(seq), 2), ["a", "b"])
        self.assertSequenceEqual(libmag.pad_seq(list(seq), 3), ["a", "b", "c"])
        self.assertSequenceEqual(
            libmag.pad_seq(list(seq), 4), ["a", "b", "c", None])
        self.assertSequenceEqual(
            libmag.pad_seq(list(seq), 4, 1), ["a", "b", "c", 1])
        self.assertSequenceEqual(
            libmag.pad_seq(list(seq), 5, [1, 2, 3, 4, 5]), ["a", "b", "c", 4, 5])
        
        # fills with last pad value since pad is shorter than seq is
        self.assertSequenceEqual(
            libmag.pad_seq(list(seq), 5, [1, 2]), ["a", "b", "c", 2, 2])
        
        # modifies the original sequence
        self.assertSequenceEqual(libmag.pad_seq(tuple(seq), 4), ["a", "b", "c", None])
        self.assertSequenceEqual(seq, ["a", "b", "c"])
        
        # modifies the original sequence
        self.assertSequenceEqual(libmag.pad_seq(seq, 4), ["a", "b", "c", None])
        self.assertSequenceEqual(seq, ["a", "b", "c", None])
    
    def test_flatten(self):
        self.assertSequenceEqual(
            list(libmag.flatten(["a", "b"])), ["a", "b"])
        self.assertSequenceEqual(
            list(libmag.flatten([["a", "b"]])), ["a", "b"])
        self.assertSequenceEqual(
            list(libmag.flatten([["a", "b"], ["c", "d"]])),
            ["a", "b", "c", "d"])
        self.assertSequenceEqual(
            list(libmag.flatten([["a", "b"], ["c", "d"], [["e", 1], "g", 3]])),
            ["a", "b", "c", "d", "e", 1, "g", 3])
    
    def test_insert_before_ext(self):
        self.assertEqual(libmag.insert_before_ext(
            "foo/bar/item.py", "totest", "_"), "foo/bar/item_totest.py")
        self.assertEqual(libmag.insert_before_ext(
            "foo/bar/item.py", "totest"), "foo/bar/itemtotest.py")
        self.assertEqual(libmag.insert_before_ext(
            "foo/bar/item", "totest", "_"), "foo/bar/item_totest")
            
    def test_splitext(self):
        self.assertEqual(
            libmag.splitext("foo/bar/item.py"), ("foo/bar/item", ".py"))
        self.assertEqual(libmag.splitext("item.py"), ("item", ".py"))
        self.assertEqual(libmag.splitext("foo/bar/item"), ("foo/bar/item", ""))
        self.assertEqual(
            libmag.splitext("foo/bar/item.file.ext"),
            ("foo/bar/item.file", ".ext"))
        self.assertEqual(libmag.splitext("foo/bar/item.file.py"), (
            "foo/bar/item.file", ".py"))
        
    def test_match_ext(self):
        self.assertEqual(
            libmag.match_ext("foo1/bar1/item1.ext1", "foo2/bar2/item2.ext2"),
            "foo2/bar2/item2.ext1")
        # if there is no extension in the path to match, it will retain its own
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
        self.assertEqual(
            libmag.get_filename_without_ext("foo/bar/item.py"), "item")
        self.assertEqual(
            libmag.get_filename_without_ext("foo/bar/item"), "item")
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
        self.assertEqual(
            libmag.splice_before("base", "file", "edit"), "baseedit")
        self.assertEqual(libmag.splice_before(
            "foo/bar/item.py", "item", "file"), "foo/bar/fileitem.py")
        self.assertEqual(libmag.splice_before(
            "foo/bar/item.py", "edit", "file"), "foo/bar/item.pyfile")
        self.assertEqual(libmag.splice_before(
            "foo/bar/item.py", ".", "file"), "foo/bar/itemfile.py")
        self.assertEqual(libmag.splice_before(
            "foo/bar/item.py", "item", "file", "/"), "foo/bar/file/item.py")
        
    def test_str_to_disp(self):
        self.assertEqual(libmag.str_to_disp("this_is_a_test"), "this is a test")
        self.assertEqual(libmag.str_to_disp(
            "       this is a test         "), "this is a test")
        self.assertEqual(
            libmag.str_to_disp("    this_is a_test    "), "this is a test")
    
    def test_format_bytes(self):
        self.assertEqual(libmag.format_bytes(10), "10 B")
        self.assertEqual(libmag.format_bytes(1024), "1.0 KB")
        self.assertEqual(libmag.format_bytes(1200), "1.2 KB")
        self.assertEqual(libmag.format_bytes(1048576), "1.0 MB")
        self.assertEqual(libmag.format_bytes(1073741824), "1.0 GB")
        self.assertEqual(libmag.format_bytes(1099511627776), "1.0 TB")
        self.assertEqual(libmag.format_bytes(1125899906842624), "1,024.0 TB")
    
    def test_make_abbreviation(self):
        name = "The long name"
        self.assertEqual(libmag.make_acronym(name), "ln")
        self.assertEqual(libmag.make_acronym(name, "o"), "Tn")
        self.assertEqual(libmag.make_acronym(name, ignore=["long"]), "Tn")
        self.assertEqual(libmag.make_acronym(name, caps=True), "LN")
        self.assertEqual(libmag.make_acronym("Short"), "Sho")
        self.assertEqual(libmag.make_acronym("Short", num_single=10), "Short")
        self.assertEqual(libmag.make_acronym(None), None)
    
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
