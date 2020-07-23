from DeepLearning.linear_unit.linear_unit import LinearUnit
from matplotlib import pyplot as plt


def get_training_dataset():
    # 随意编写 5 个数据，特征向量为 5 个人的工作时间
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 标签为 5 个人对应的工资
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    # 使用线性单元模型，输入参数为工作年限，参数数目为 1
    lu = LinearUnit(1)
    input_vecs, labels = get_training_dataset()
    # 训练迭代 10 次，学习率为 0.01
    lu.train(input_vecs, labels, 10, 0.01)
    return lu


def plot(linear_unit):
    input_vecs, labels = get_training_dataset()
    # figure() 创建一个 Figure 对象，与用户交互的整个窗口，这个 figure 中容纳着 subplots
    fig = plt.figure()
    # 在 figure 对象中创建 1 行 1 列的第 1 个图
    ax = fig.add_subplot(111)
    # scatter(x, y) 绘制散点图，其中的 x，y 是相同长度的数组序列
    ax.scatter(list(map(lambda x: x[0], input_vecs)), labels)
    # 设置权重
    weights = linear_unit.weights
    # 设置偏置项
    bias = linear_unit.bias
    y1 = 0 * weights[0] + bias
    y2 = 12 * weights[0] + bias
    plt.plot([0, 12], [y1, y2])
    plt.show()


def main():
    # 训练线性单元模型
    linear_unit = train_linear_unit()
    # 打印训练获得的权重和偏置
    print(linear_unit)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    plot(linear_unit)


if __name__ == '__main__':
    main()
