import unittest
import numpy as np
from investigate import dataset,matched_width,count_params,paired_summary,SPLITS,GROUPS

class StudyDesign(unittest.TestCase):
    def test_nested_and_fixed_splits(self):
        small=dataset(0,32,16,16,4);large=dataset(0,64,16,16,4)
        np.testing.assert_array_equal(small[0],large[0][:32])
        np.testing.assert_array_equal(small[1],large[1])
        for split in SPLITS:np.testing.assert_array_equal(small[2][split],large[2][split])
        self.assertFalse(np.array_equal(small[1],small[2]['iid']))
        self.assertFalse(np.array_equal(small[0],dataset(1,32,16,16,4)[0]))
    def test_nearest_budget(self):
        for k in [2,3,6]:
            h=matched_width(16,64,k);error=abs(k*count_params(16,h,1)-count_params(16,64,k))
            self.assertEqual(error,min(abs(k*count_params(16,j,1)-count_params(16,64,k)) for j in range(1,65)))
    def test_paired_aggregation(self):
        rows=[]
        for split in SPLITS:
            for group,ids in GROUPS.items():
                for seed in [0,1]:
                    for task in ids:
                        for method in ['shared','matched','independent','ridge']:
                            value=1 if method=='shared' else (2 if seed==0 else 4)
                            rows.append(dict(train=32,split=split,group=group,seed=seed,method=method,nrmse=value))
        for r in paired_summary(rows):
            self.assertEqual(r['gain'],2)
            self.assertAlmostEqual(r['gain_seed_sd'],np.sqrt(2))
            self.assertEqual(r['positive_seeds'],2)

if __name__=='__main__':unittest.main()
