import pkg_resources

resource_package = __name__


def load_datingTestSet2():
    filename = pkg_resources.resource_filename(resource_package, 'data/datingTestSet2.txt')
    return filename


def load_testDigits(file_name_str):
    filename = pkg_resources.resource_filename(__name__, 'data/testDigits/%s' % file_name_str)
    return filename


def load_trainingDigits(file_name_str):
    filename = pkg_resources.resource_filename(__name__, 'data/trainingDigits/%s' % file_name_str)
    return filename


def load_trainingDigits_list():
    return pkg_resources.resource_listdir(__name__, 'data/trainingDigits')


def load_testDigits_list():
    return pkg_resources.resource_listdir(__name__, 'data/testDigits')
