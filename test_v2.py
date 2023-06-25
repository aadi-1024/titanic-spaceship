import unittest
import numpy as np

from v2 import CabinFeatures

class FeaturesTest(unittest.TestCase):
    def test_cabin(self):
        cf = CabinFeatures()
        a = ['A/1/P', 'B/2/S']
        b = np.array([['A/1/P', 'A', 'P'], ['B/2/S', 'B', 'S']])
        self.assertTrue((cf.fit_transform(a)== b).all())

if __name__ == '__main__':
    unittest.main(verbosity=2)