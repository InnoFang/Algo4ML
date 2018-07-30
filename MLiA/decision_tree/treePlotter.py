import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

# 定义文本框和箭头格式
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrow_args)


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    # # plot test code
    # plotNode(u'决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    # plotNode(u'叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def plotTree(myTree, parentPt, nodeTxt):
    num_leafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    first_str = list(myTree)[0]
    cntr_pt = (plotTree.xOff + (1.0 + float(num_leafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntr_pt, parentPt, nodeTxt)
    plotNode(first_str, cntr_pt, parentPt, decision_node)
    second_dict = myTree[first_str]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plotTree(second_dict[key], cntr_pt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(second_dict[key], (plotTree.xOff, plotTree.yOff), cntr_pt, leaf_node)
            plotMidText((plotTree.xOff, plotTree.yOff), cntr_pt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def plotMidText(cntrPt, parentPt, txtString):
    x_mid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    y_mid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(x_mid, y_mid, txtString)


def getNumLeafs(myTree):
    """
    获取叶节点的数目
    :param myTree: 
    :return: 
    """
    num_leafs = 0
    # first_str = list(myTree.keys())[0]
    first_str = list(myTree)[0]
    second_dict = myTree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += getNumLeafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def getTreeDepth(myTree):
    """
    获取树的层数
    :param myTree: 
    :return: 
    """
    max_depth = 0
    # first_str = list(myTree.keys())[0]
    first_str = list(myTree)[0]
    second_dict = myTree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + getTreeDepth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieveTree(i):
    """
    预先存储的树信息，避免每次测试代码时都要从数据中创建树
    :param i: 
    :return: 
    """
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}}}}}
                     ]
    return list_of_trees[i]
