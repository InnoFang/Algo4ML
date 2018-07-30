from MLiA.decision_tree import trees, treePlotter, dataset
import unittest


class TestDeciseTree(unittest.TestCase):

    def test_createDataSet(self):
        # 获得数据集和标签
        my_data, labels = trees.createDataSet()
        print(my_data)
        # 计算数据集的信息熵
        print(trees.calcShannonEnt(my_data))

    def test_calcShannonEnt(self):
        my_data, labels = trees.createDataSet()
        # 增加一个新的分类 'maybe' ，观察信息熵的变化
        my_data[0][-1] = 'maybe'
        print(trees.calcShannonEnt(my_data))
        # 结论：混合的数据越多，信息熵越高

    def test_extend_and_append(self):
        # extend 和 append 的区别
        a = [1, 2, 3]
        b = [4, 5, 6]
        # 使用 extend 会将原列表中的值添加进目的列表
        a.extend(b)
        print(a)
        # 使用 append 会把整个列表当作元素添加进目的列表
        a.append(b)
        print(a)

    def test_splitDataSet(self):
        my_data, labels = trees.createDataSet()
        print(my_data)
        print(trees.splitDataSet(my_data, 0, 1))
        print(trees.splitDataSet(my_data, 0, 0))

    def test_chooseBestFeatureToSplit(self):
        my_data, labels = trees.createDataSet()
        print(trees.chooseBestFeatureToSplit(my_data))
        print(my_data)

    def test_createTree(self):
        my_data, labels = trees.createDataSet()
        my_tree = trees.createTree(my_data, labels)
        print(my_tree)

    def test_retrieveTree(self):
        print(treePlotter.retrieveTree(0))
        print(treePlotter.retrieveTree(1))

    def test_createPlot(self):
        my_tree = treePlotter.retrieveTree(0)
        print(treePlotter.getNumLeafs(my_tree))
        print(treePlotter.getTreeDepth(my_tree))
        treePlotter.createPlot(my_tree)

    def test_createPlot_after_adding_data(self):
        # 添加数据，重新绘制树形图观察输出结果的变化
        my_tree = treePlotter.retrieveTree(0)
        my_tree['no surfacing'][3] = 'maybe'
        print(my_tree)
        treePlotter.createPlot(my_tree)

    def test_classify(self):
        my_data, labels = trees.createDataSet()
        my_tree = treePlotter.retrieveTree(0)
        print(trees.classify(my_tree, labels, [1, 0]))
        print(trees.classify(my_tree, labels, [1, 1]))

    def test_storeTree_and_grabTree(self):
        my_tree = treePlotter.retrieveTree(0)
        trees.storeTree(my_tree, 'classifierStorage.txt')
        print(trees.grabTree('classifierStorage.txt'))

    def test_predict_lenses_type(self):
        fr = dataset.load_lenses()
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lenses_tree = trees.createTree(lenses, lenses_labels.copy())
        print(lenses_tree)
        print(trees.classify(lenses_tree, lenses_labels, ['young', 'hyper', 'no', 'reduced']))
        treePlotter.createPlot(lenses_tree)