from __future__ import annotations
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize import classification,flags,descriptive


def endpoint(effects,group_sd):
    return {'effects':effects,'paired_mean':sum(effects)/len(effects),'B_sample_SD':group_sd,'C_sample_SD':group_sd}


class StatisticalUnitTests(unittest.TestCase):
    def test_direction_reversals_cannot_be_robust(self):
        self.assertEqual(classification(endpoint([.01,-.02,.03],.001),endpoint([1,-1,2],.01)),'INCONCLUSIVE')

    def test_same_direction_below_training_variation_is_inconclusive(self):
        self.assertEqual(classification(endpoint([.001,.002,.003],.05),endpoint([.01,.02,.03],.4)),'INCONCLUSIVE')

    def test_raw_foreground_disagreement_is_mixed(self):
        self.assertEqual(classification(endpoint([.1,.2,.3],.01),endpoint([-.1,-.2,-.3],.01)),'MIXED')

    def test_prospective_robust_rule_checks_both_endpoints(self):
        for sign,label in [(1,'ROBUST_POSITIVE'),(-1,'ROBUST_NEGATIVE')]:
            self.assertEqual(classification(endpoint([sign*x for x in [.1,.2,.3]],.01),endpoint([sign*x for x in [1,2,3]],.1)),label)

    def test_sample_sd_uses_three_training_repeats(self):
        result=descriptive([1.,2.,3.]);self.assertEqual(result['sample_SD'],1.)
        self.assertEqual(result['mean'],2.)
        self.assertEqual(result['min'],1.);self.assertEqual(result['max'],3.)

    def test_descriptive_flag_retains_unusual_seed(self):
        self.assertIn(2,flags([0.,.01,1.]))
        values=[0.,.01,1.];flags(values);self.assertEqual(len(values),3)

if __name__=='__main__':unittest.main()
