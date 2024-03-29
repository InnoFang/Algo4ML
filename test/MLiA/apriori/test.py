import unittest
from MLiA.apriori import apriori


class TestApriori(unittest.TestCase):
    def test_apriori(self):
        dataset = apriori.load_dataset()
        print('dataset: ', dataset)

        l1, support_data = apriori.apriori(dataset, min_support=0.7)
        print('L(0.7):', l1)
        print('support_data(0.7):', support_data)

        print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')

        l2, support_data = apriori.apriori(dataset, min_support=0.7)
        print('L(0.5):', l2)
        print('support_data(0.5):', support_data)

    def test_generate_rules(self):
        dataset = apriori.load_dataset()
        print('dataset: ', dataset)

        l1, support_data = apriori.apriori(dataset, min_support=0.5)
        print('L(0.5):', l1)
        print('support_data(0.5):', support_data)

        rules = apriori.generate_rules(l1, support_data, min_conf=0.5)
        print('rules:', rules)
