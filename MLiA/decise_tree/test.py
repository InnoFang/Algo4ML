from decise_tree import treePlotter

from MLiA.decise_tree import trees

# 获得数据集和标签
my_data, labels = trees.createDataSet()
print(my_data)

# 计算数据集的信息熵
# print(trees.calcShannonEnt(my_data))


# 增加一个新的分类 'maybe' ，观察信息熵的变化
# my_data[0][-1] = 'maybe'
# print(trees.calcShannonEnt(my_data))
# 结论：混合的数据越多，信息熵越高


# # extend 和 append 的区别
# a = [1, 2, 3]
# b = [4, 5, 6]
# # 使用 extend 会将原列表中的值添加进目的列表
# a.extend(b)
# print(a)
# # 使用 append 会把整个列表当作元素添加进目的列表
# a.append(b)
# print(a)


# print(trees.splitDataSet(my_data, 0, 1))
# print(trees.splitDataSet(my_data, 0, 0))


# print(trees.chooseBestFeatureToSplit(my_data))
# print(my_data)


# my_tree = trees.createTree(my_data, labels)
# print(my_tree)

# treePlotter.createPlot()

# print(treePlotter.retrieveTree(0))
# print(treePlotter.retrieveTree(1))
my_tree = treePlotter.retrieveTree(0)
# print(treePlotter.getNumLeafs(my_tree))
# print(treePlotter.getTreeDepth(my_tree))
# treePlotter.createPlot(my_tree)

# # 添加数据，重新绘制树形图观察输出结果的变化
# my_tree['no surfacing'][3] = 'maybe'
# print(my_tree)
# treePlotter.createPlot(my_tree)

# print(trees.classify(my_tree, labels, [1, 0]))
# print(trees.classify(my_tree, labels, [1, 1]))

# trees.storeTree(my_tree, 'data/classifierStorage.txt')
# print(trees.grabTree('data/classifierStorage.txt'))


fr = open('data/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
lenses_tree = trees.createTree(lenses, lenses_labels.copy())
print(lenses_tree)
print(trees.classify(lenses_tree, lenses_labels, ['young', 'hyper', 'no', 'reduced']))
treePlotter.createPlot(lenses_tree)
