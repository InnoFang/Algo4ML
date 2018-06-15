from MLiA.svm import svm

dataArr, labelArr = svm.loadDataSet('data/testSet.txt')
print(labelArr)