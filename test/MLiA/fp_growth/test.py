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

    def test_find_prefix_paths(self):
        simple_data = fp_growth.load_simple_data()
        init_set = fp_growth.create_init_set(simple_data)
        fp_tree, header_tab = fp_growth.create_tree(init_set, 3)
        print(fp_growth.find_prefix_path('x', header_tab['x'][1]))
        print(fp_growth.find_prefix_path('z', header_tab['z'][1]))
        print(fp_growth.find_prefix_path('r', header_tab['r'][1]))

    def test_create_cond_tree(self):
        simple_data = fp_growth.load_simple_data()
        init_set = fp_growth.create_init_set(simple_data)
        fp_tree, header_tab = fp_growth.create_tree(init_set, 3)
        freq_items = []
        fp_growth.mine_tree(fp_tree, header_tab, 3, set([]), freq_items)
        print(freq_items)
