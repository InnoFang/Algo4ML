from MLiA.knn.kNN import file2matrix
import pkg_resources

resource_package = __name__

def load_datingTestSet2():
    filename = pkg_resources.resource_filename(resource_package, 'data/datingTestSet2.txt')
    return filename
