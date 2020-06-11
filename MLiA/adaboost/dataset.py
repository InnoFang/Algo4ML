import pkg_resources

resource_package = __name__


def loadDataSet(fileName):
    num_feat = len(open(fileName).readline().split('\t'))
    data_mat, label_mat = [], []
    fr = open(fileName)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def load_horseColicTest():
    filename = pkg_resources.resource_filename(resource_package, 'data/horseColicTest2.txt')
    return loadDataSet(filename)


def load_horseColicTraining():
    filename = pkg_resources.resource_filename(resource_package, 'data/horseColicTraining2.txt')
    return loadDataSet(filename)
