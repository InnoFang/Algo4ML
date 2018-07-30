from thoughtfulML.k_nearest_neighbors.regression import RegressionTest
from thoughtfulML.k_nearest_neighbors.dataset import load_king_county_data_geocoded
import unittest

class testKNearestNeighbors(unittest.TestCase):

    def test_plot_error_rates(self):
        regression_test = RegressionTest()
        regression_test.load_csv_file(load_king_county_data_geocoded(), 100)
        regression_test.plot_error_rates()
