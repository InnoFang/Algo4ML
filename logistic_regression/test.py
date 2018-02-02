from logistic_regression import logRegres
import numpy as np

data_arr, label_mat = logRegres.loadDataSet()
# print(logRegres.gradAscent(data_arr, label_mat))

# #Test for gradAscent
# weights = logRegres.gradAscent(data_arr, label_mat)
# # getA equivalent to ``np.asarray(self)``.
# # Change the weights' type into array type
# logRegres.plotBestFit(weights.getA())


# # Test for stocGradAscent0
# weights = logRegres.stocGradAscent0(np.array(data_arr), label_mat)
# logRegres.plotBestFit(weights)


# # Test for stocGradAscent1
# weights = logRegres.stocGradAscent1(np.array(data_arr), label_mat)
# logRegres.plotBestFit(weights)
# weights = logRegres.stocGradAscent1(np.array(data_arr), label_mat, 500)
# logRegres.plotBestFit(weights)


# Predict the death rate of a sick horse from a hernia (or colic) disease
logRegres.multiTest()


