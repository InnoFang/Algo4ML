import pkg_resources

resource_package = __name__


def loadDataSet(fileName):
    num_feat = len(open(fileName).readline().split('\t')) - 1
    data_mat, label_mat = [], []
    fr = open(fileName)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat
