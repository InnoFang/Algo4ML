from MLiA.svm import svm

dataArr, labelArr = svm.loadDataSet('data/testSet.txt')
# print(labelArr)

b, alphas = svm.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print(b)