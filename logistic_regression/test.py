from logistic_regression import logRegres

data_arr, label_mat = logRegres.loadDataSet()
print(logRegres.gradAscent(data_arr, label_mat))