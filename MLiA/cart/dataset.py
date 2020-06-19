import pkg_resources

resource_package = __name__


def loadDataSet(fileName):
    data_mat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            # 将每行映射成浮点数
            flt_line = []
            for i in cur_line:
                flt_line.append(float(i))
            data_mat.append(flt_line)
    return data_mat


def load_ex00():
    filename = pkg_resources.resource_filename(resource_package, 'data/ex00.txt')
    return loadDataSet(filename)


def load_ex2():
    filename = pkg_resources.resource_filename(resource_package, 'data/ex2.txt')
    return loadDataSet(filename)


def load_ex2test():
    filename = pkg_resources.resource_filename(resource_package, 'data/ex2test.txt')
    return loadDataSet(filename)


def load_exp2():
    filename = pkg_resources.resource_filename(resource_package, 'data/exp2.txt')
    return loadDataSet(filename)


def load_sine():
    filename = pkg_resources.resource_filename(resource_package, 'data/sine.txt')
    return loadDataSet(filename)
