import unittest
import numpy as np

from v2 import CabinFeatures, PidFeatures


class FeaturesTest(unittest.TestCase):
    def test_cabin(self):
        cf = CabinFeatures()
        a = ['A/1/P', 'B/2/S']
        b = np.array([['A', 'P'], ['B', 'S']])
        self.assertTrue((cf.fit_transform(a) == b).all())

    def test_pid(self):
        pf = PidFeatures()
        a = ['001_01', '003_05']
        b = np.array(['01', '05'])
        self.assertTrue((pf.fit_transform(a) == b).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
