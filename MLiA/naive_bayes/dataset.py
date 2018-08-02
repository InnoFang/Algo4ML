import pkg_resources

resource_package = __name__


def load_ham(num):
    filename = pkg_resources.resource_filename(resource_package, 'data/ham/%s.txt' % num)
    return filename

def load_spam(num):
    filename = pkg_resources.resource_filename(resource_package, 'data/spam/%s.txt' % num)
    return filename