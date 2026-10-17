"""Comprehensive Verification Test Suite for Laboratory-Ready PLQ Modules.

Conforming to Nuclear Software Quality Standards: ASME NQA-1 Subpart 2.7, IEC 60880.
"""

from fractions import Fraction
import os
import sys
import unittest
from pathlib import Path

# Add PLQ and topos-lang to sys.path
plq_root = str(Path(__file__).parent.parent.resolve())
if plq_root not in sys.path:
    sys.path.insert(0, plq_root)

for candidate in (
    str(Path(__file__).parent.parent / "topos-lang" / "src"),
    "/app/PLQ/topos-lang/src",
    "/app/topos-lang/src",
    "/home/eliasbrahim/topos-lang/src",
):
    if Path(candidate).exists() and candidate not in sys.path:
        sys.path.insert(0, candidate)

from plq.dacc.complex import DACCComplex
from plq.dacc.covalent_anchor import CovalentStaplingEngine
from plq.wetlab.spps_filter import SPPSManufacturabilityScanner
from plq.wetlab.solubility import AqueousSolubilityAnalyzer
from plq.wetlab.desolvation import ContinuumDesolvationEngine
from plq.wetlab.immunogenicity import ImmunogenicityAnalyzer
from plq.wetlab.lab_dossier import LabDossierGenerator
from plq.gnc.ramachandran import RamachandranSheafEvaluator
from plq.gnc.steric_clash import CGAStericClashDetector
from plq.sheaf.cohomology import CellularSheafEngine
from plq.polypharmacology.selectivity import TopologicalSelectivityEvaluator
from plq.nuclear.qa_quorum import NuclearQAQuorumEngine


class TestLaboratoryReadyPLQ(unittest.TestCase):
    def test_spps_filter_detects_aggregation_and_liabilities(self):
        # 1. Sequence with aggregation and aspartimide
        bad_seq = "ACVVVDEFGDGKL"
        report_bad = SPPSManufacturabilityScanner.scan_sequence(bad_seq)
        self.assertIn("VVV", report_bad.hydrophobic_runs)
        self.assertGreater(len(report_bad.aspartimide_traps), 0)
        self.assertFalse(report_bad.is_synthesizable)
        self.assertLess(report_bad.spps_score, 0.80)

        # 2. Clean, synthesizable sequence
        clean_seq = "EVQLVESGGGLVQPG"
        report_clean = SPPSManufacturabilityScanner.scan_sequence(clean_seq)
        self.assertEqual(len(report_clean.hydrophobic_runs), 0)
        self.assertEqual(len(report_clean.aspartimide_traps), 0)
        self.assertTrue(report_clean.is_synthesizable)
        self.assertGreaterEqual(report_clean.spps_score, 0.80)

    def test_solubility_gravy_and_pI(self):
        # Hydrophilic sequence
        seq = "EEDDKKRHEE"
        prof = AqueousSolubilityAnalyzer.evaluate_solubility(seq)
        self.assertLess(prof.gravy_score, -1.0)
        self.assertTrue(prof.is_soluble_in_pbs)
        self.assertGreater(prof.isoelectric_point_pI, 3.0)
        self.assertLess(prof.isoelectric_point_pI, 7.0)

    def test_covalent_disulfide_anchor(self):
        c = DACCComplex("DisulfideLoop")
        c.add_0cell("cys_1", index=1)
        c.add_0cell("res_2", index=2)
        c.add_0cell("res_3", index=3)
        c.add_0cell("cys_4", index=4)
        c.add_1cell("bb_1_2", "cys_1", "res_2")
        c.add_1cell("bb_2_3", "res_2", "res_3")
        c.add_1cell("bb_3_4", "res_3", "cys_4")

        report = CovalentStaplingEngine.add_disulfide_bridge(c, "cys_1", "cys_4")
        self.assertTrue(report.is_nilpotent)
        self.assertEqual(report.loop_size, 3)
        self.assertGreater(report.proteolytic_stability_hours, 1.0)
        self.assertTrue(c.verify_all_nilpotence())

    def test_ramachandran_torsion_sheaf(self):
        seq = "AAAAAAAAAA"
        report = RamachandranSheafEvaluator.evaluate_chain(seq)
        self.assertTrue(report.is_stereochemically_valid)
        self.assertGreaterEqual(report.percent_favored, 90.0)

    def test_steric_clash_detection(self):
        # Two carbon atoms at 3.5 A (no clash)
        atoms_no_clash = [
            ("C1", "C", (0.0, 0.0, 0.0)),
            ("C2", "C", (3.5, 0.0, 0.0)),
        ]
        rep_ok = CGAStericClashDetector.evaluate_pairwise_clashes(atoms_no_clash)
        self.assertTrue(rep_ok.is_clash_free)
        self.assertEqual(rep_ok.clash_count, 0)

        # Two carbon atoms at 1.0 A (severe clash)
        atoms_clash = [
            ("C1", "C", (0.0, 0.0, 0.0)),
            ("C2", "C", (1.0, 0.0, 0.0)),
        ]
        rep_bad = CGAStericClashDetector.evaluate_pairwise_clashes(atoms_clash)
        self.assertFalse(rep_bad.is_clash_free)
        self.assertGreater(rep_bad.clash_count, 0)

    def test_sheaf_cohomology_obstruction(self):
        c = DACCComplex("TriangleFace")
        c.add_0cell("v0")
        c.add_0cell("v1")
        c.add_0cell("v2")
        c.add_1cell("e01", "v0", "v1")
        c.add_1cell("e12", "v1", "v2")
        c.add_1cell("e02", "v0", "v2")
        c.add_2cell("f0", [("e01", 1), ("e12", 1), ("e02", -1)])

        rep = CellularSheafEngine.compute_obstruction_cohomology(c)
        self.assertTrue(rep.is_globally_unobstructed)
        self.assertEqual(rep.dim_H1_obstructions, 0)

    def test_desolvation_engine(self):
        contacts = [(+1, -1, 3.2)] # Salt bridge at 3.2 A
        rep = ContinuumDesolvationEngine.evaluate_interface_desolvation(contacts, buried_hydrophobic_residues=3)
        self.assertLess(rep.coulombic_energy_kcal_mol, 0.0)
        self.assertGreater(rep.desolvation_penalty_kcal_mol, 0.0)
        self.assertTrue(rep.is_thermodynamically_favorable)

    def test_immunogenicity_and_humanness(self):
        human_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV"
        rep = ImmunogenicityAnalyzer.evaluate_immunogenicity(human_seq)
        self.assertGreaterEqual(rep.oasis_humanness_score, 0.70)
        self.assertTrue(rep.is_immunogenically_acceptable)

    def test_selectivity_index(self):
        evaluator = TopologicalSelectivityEvaluator()
        rep = evaluator.evaluate_selectivity("HEHDHAALLL")
        self.assertGreater(rep.topological_selectivity_index, 0.0)
        self.assertTrue(rep.is_cross_species_active)

    def test_nuclear_qa_quorum_and_lab_dossier(self):
        cert = NuclearQAQuorumEngine.evaluate_candidate(
            candidate_name="TOPOS_TEST_01",
            sequence="EVQLVESGGGLVQPG",
            is_nilpotent=True,
            beta_1=0,
            dim_h1_obstruction=0,
            rmsd_angstrom=1.2,
            clash_count=0,
            ramachandran_favored_pct=95.0,
            spps_score=0.92,
            gravy_score=-0.25,
            aspartimide_count=0,
            mhc2_high_affinity_count=0,
            nominal_delta_g=-9.2,
            uncertainty_sigma=1.0,
        )
        self.assertEqual(cert.quorum_decision, "CERTIFIED_LAB_READY")
        self.assertTrue(cert.is_laboratory_ready)
        self.assertEqual(cert.votes_passed, 3)

        # Lab dossier tests
        ms = LabDossierGenerator.calculate_mass_envelope("EVQLVESGGGLVQPG")
        self.assertGreater(ms.monoisotopic_mass_da, 1000.0)
        pdb_str = LabDossierGenerator.generate_pdb_string("EVQLVESGGGLVQPG")
        self.assertIn("ATOM", pdb_str)
        sop_str = LabDossierGenerator.generate_synthesis_sop("EVQLVESGGGLVQPG")
        self.assertIn("STANDARD OPERATING PROCEDURE", sop_str)


if __name__ == "__main__":
    unittest.main()
