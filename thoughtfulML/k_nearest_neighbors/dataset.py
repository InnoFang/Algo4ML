import pkg_resources

resource_package = __name__


def load_king_county_data_geocoded():
    filename = pkg_resources.resource_filename(resource_package, 'data/king_county_data_geocoded.csv')
    return filename
