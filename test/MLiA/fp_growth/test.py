from MLiA.fp_growth import fp_growth
import unittest


class TestFPGrowth(unittest.TestCase):
    def test_TreeNode(self):
        root = fp_growth.TreeNode('pyramid', 9, None)
        root.children['eye'] = fp_growth.TreeNode('eye', 13, None)
        root.disp()

        root.children['phoenix'] = fp_growth.TreeNode('phoenix', 3, None)
        root.disp()

    def test_create_tree(self):
        simple_data = fp_growth.load_simple_data()
        init_set = fp_growth.create_init_set(simple_data)
        fp_tree, header_tab = fp_growth.create_tree(init_set, 3)
        fp_tree.disp()
