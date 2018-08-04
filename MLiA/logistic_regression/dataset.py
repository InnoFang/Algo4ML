import pkg_resources

resource_package = __name__


def load_horseColicTest():
    filename = pkg_resources.resource_filename(resource_package, 'data/horseColicTest.txt')
    return filename


def load_horseColicTraining():
    filename = pkg_resources.resource_filename(resource_package, 'data/horseColicTraining.txt')
    return filename


def load_testSet():
    filename = pkg_resources.resource_filename(resource_package, 'data/testSet.txt')
    return filename
