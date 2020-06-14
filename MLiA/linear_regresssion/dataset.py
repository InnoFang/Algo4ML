import pkg_resources

resource_package = __name__


def loadDataSet(fileName):
    num_feat = 0
    with open(fileName) as fr:
        num_feat = len(fr.readline().split('\t')) - 1

    data_mat, label_mat = [], []
    with open(fileName) as fr:
        for line in fr.readlines():
            line_arr = []
            cur_line = line.strip().split('\t')
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))
        return data_mat, label_mat


def load_ex0():
    filename = pkg_resources.resource_filename(resource_package, 'data/ex0.txt')
    return loadDataSet(filename)


def load_ex1():
    filename = pkg_resources.resource_filename(resource_package, 'data/ex1.txt')
    return loadDataSet(filename)


def load_abalone():
    filename = pkg_resources.resource_filename(resource_package, 'data/abalone.txt')
    return loadDataSet(filename)
