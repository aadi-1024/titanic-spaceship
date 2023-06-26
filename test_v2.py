import unittest
import numpy as np
import pandas as pd

from v2 import CabinFeatures, PidFeatures, MakeDataFrame


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

    def test_makedf(self):
        md = MakeDataFrame(columns=['a', 'b', 'c'])
        a = np.array([0, 1, 2, 3, 4, 5])
        a.shape = (2, 3)
        b = pd.DataFrame(a, columns=['a', 'b', 'c'])
        self.assertTrue(md.fit_transform(a).equals(b))

if __name__ == '__main__':
    unittest.main(verbosity=2)
