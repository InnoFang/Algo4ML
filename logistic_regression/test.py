from logistic_regression import logRegres
import numpy as np

data_arr, label_mat = logRegres.loadDataSet()
print(logRegres.gradAscent(data_arr, label_mat))

weights = logRegres.gradAscent(data_arr, label_mat)
# getA equivalent to ``np.asarray(self)``.
# Change the weights' type into array type
logRegres.plotBestFit(weights.getA())
