import pkg_resources

resource_package = __name__


def load_lenses():
    filename = pkg_resources.resource_filename(resource_package, 'data/lenses.txt')
    return open(filename)
