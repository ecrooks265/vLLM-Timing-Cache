import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('analysis', Path(__file__).resolve().parents[1] / 'scripts/analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def test_auc_direction(self):
        self.assertEqual(analysis.auc_ap([1,1,0,0], [1,2,3,4]), (1.,1.))

    def test_equal_scores_do_not_fake_precision(self):
        self.assertEqual(analysis.auc_ap([1,0,1,0], [10,10,10,10]), (.5,.5))

    def test_incomplete_image_pair_rejected(self):
        with self.assertRaises(ValueError):
            analysis.paired([{'rep':0,'image_id':'a','victim_primed':1}])

    def test_bootstrap_preserves_image_pairs(self):
        rows = []
        for rep in range(3):
            for i in range(4):
                for label in [0,1]:
                    rows.append({'rep':rep,'image_id':str(i),'victim_primed':label,
                                 'first_content_ms':(10 if label else 30)+rep+i})
        ci = analysis.boot_ci(rows, 30)
        self.assertEqual(ci['roc_auc'], [1.,1.])
        self.assertEqual(ci['paired_delta_ms'], [20.,20.])


if __name__ == '__main__':
    unittest.main()
