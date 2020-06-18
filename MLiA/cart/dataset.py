import pkg_resources

resource_package = __name__


def loadDataSet(fileName):
    data_mat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            # 将每行映射成浮点数
            flt_line = map(float, cur_line)
            data_mat.append(flt_line)
    return data_mat


def load_ex00():
    filename = pkg_resources.resource_filename(resource_package, 'data/ex00.txt')
    return loadDataSet(filename)
