from decise_tree import trees

# 获得数据集和标签
my_data, labels = trees.createDataSet()
print(my_data)

# 计算数据集的信息熵
print(trees.calcShannonEnt(my_data))

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


print(trees.splitDataSet(my_data, 0, 1))
print(trees.splitDataSet(my_data, 0, 0))