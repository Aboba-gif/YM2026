"""Проверки знака пары, исключений, MCSE и неизменности исходных записей."""

from tests.experiments.source_recovery.analysis.fixtures.campaign import CampaignFixture
import json
from pathlib import Path
import tempfile
import unittest

from experiments.source_recovery.analysis import summary as s


class ScientificFixtures(CampaignFixture, unittest.TestCase):


    """Проверки парного анализатора на записях CampaignFixture."""

    def test_exclusions_pending_and_pair_sign_are_visible(self):
        before={str(p):s.digest(p) for p in self.root.rglob("*.json")}
        report=s.summarize(self.root,"main")
        pair=report["primary_paired_results"][0]
        self.assertEqual((pair["nplanned"],pair["nactual"],pair["nexcluded"]),(3,1,2))
        self.assertAlmostEqual(pair["statistics"]["E_q"]["mean"],-.1)
        self.assertIsNone(pair["statistics"]["E_q"]["mcse"])
        self.assertEqual(len(pair["all_scored_pairs_with_diagnostic_flags"]),2)
        self.assertIn("right:alpha_boundary_unresolved",pair["exclusion_counts"])
        self.assertEqual(report["coverage"]["nominal_candidates"],150)
        self.assertEqual(report["coverage"]["candidate_attempted"],100)
        self.assertEqual(before,{str(p):s.digest(p) for p in self.root.rglob("*.json")})

    def test_sd_mcse_are_computed_on_paired_differences(self):
        values=[-2.,0.,2.]
        result=s.moments(values)
        self.assertEqual(result["mean"],0)
        self.assertEqual(result["sd"],2)
        self.assertAlmostEqual(result["mcse"],2/(3**.5))

    def test_seal_and_binding_tampering_are_rejected(self):
        p=self.root/"PG10/replicate_1.json"
        cp,_=s.snapshot(p)
        cp["paths"]["main/H1"]["procedure_accepted"]=False
        self.write("PG10/replicate_1.json",cp)
        with self.assertRaisesRegex(ValueError,"seal"):
            s.summarize(self.root,"main")


if __name__=="__main__":
    unittest.main()
