"""Comprehensive Unit and Integration Test Suite for 8D HCST and ToposLang 2.0 PLQ Upgrades."""

import math
import sys
import unittest
from pathlib import Path

# Add PLQ and topos-lang to sys.path
plq_root = str(Path(__file__).parent.parent.resolve())
if plq_root not in sys.path:
    sys.path.insert(0, plq_root)

topos_path = str(Path(__file__).parent.parent / "topos-lang" / "src")
if Path(topos_path).exists() and topos_path not in sys.path:
    sys.path.insert(0, topos_path)

from plq.dacc.complex import DACCComplex, NilpotencyViolationError
from plq.dacc.fingerprint import TopologicalDACCInvariant, TopologicalSignature
from plq.gnc.clifford import Clifford8D, Multivector
from plq.gnc.steric_clash import CGAStericClashDetector
from plq.hodge.laplacian import CombinatorialHodgeLaplacian
from plq.hodge.engine import HodgeGuidedGenerator
from plq.io.pdb_parser import PDBParser
from plq.neural.hcst_pocket import PLQHCST8DTransformer
from plq.polypharmacology.dual_tier import DualTierScheduler
from plq.wetlab.desolvation import ContinuumDesolvationEngine
from plq.compiler.topos_bridge import ToposContractBridge


class Test8DDACCAndNilpotency(unittest.TestCase):
    def test_3d_and_4d_hypercube_nilpotency(self):
        c3 = DACCComplex.create_nd_hypercube(3)
        self.assertTrue(c3.verify_all_nilpotence())
        self.assertEqual(len(c3.cells[0]), 8)
        self.assertEqual(len(c3.cells[1]), 12)
        self.assertEqual(len(c3.cells[2]), 6)
        self.assertEqual(len(c3.cells[3]), 1)

        c4 = DACCComplex.create_nd_hypercube(4)
        self.assertTrue(c4.verify_all_nilpotence())
        self.assertTrue(c4.verify_all_nilpotence_8d())
        self.assertEqual(len(c4.cells[0]), 16)
        self.assertEqual(len(c4.cells[4]), 1)

    def test_higher_cell_nilpotency_enforcement(self):
        c = DACCComplex("NilpotencyTest")
        c.add_0cell("v0")
        c.add_0cell("v1")
        c.add_1cell("e01", "v0", "v1")
        # Trying to add a 2-cell with an open boundary (e01 only) must raise NilpotencyViolationError
        with self.assertRaises(NilpotencyViolationError):
            c.add_2cell("f_open", [("e01", 1)], verify_nilpotency=True)

    def test_8d_euler_characteristic(self):
        c4 = DACCComplex.create_nd_hypercube(4)
        # For an n-ball/hypercube, chi = 1
        self.assertEqual(c4.euler_characteristic(), 1)


class Test256BladeCliffordAlgebra(unittest.TestCase):
    def test_binomial_grade_dimensions(self):
        expected = (1, 8, 28, 56, 70, 56, 28, 8, 1)
        self.assertEqual(Clifford8D.GRADE_DIMENSIONS, expected)
        self.assertEqual(sum(expected), 256)

    def test_8d_vector_embedding_and_norm(self):
        coords = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        v = Clifford8D.vector_8d(coords)
        expected_norm = math.sqrt(sum(x * x for x in coords))
        self.assertAlmostEqual(v.norm(), expected_norm, places=4)

    def test_8d_wedge_antisymmetry(self):
        v1 = Clifford8D.vector_8d((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        v2 = Clifford8D.vector_8d((0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        w12 = v1.wedge(v2)
        w21 = v2.wedge(v1)
        # v1 ^ v2 == -(v2 ^ v1)
        for b, c in w12.components.items():
            self.assertAlmostEqual(c, -w21.components[b], places=5)

    def test_8d_dual_stream_attention_score(self):
        v1 = Clifford8D.vector_8d((1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        v2 = Clifford8D.vector_8d((1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        # Parallel vectors: wedge is zero, alignment is positive
        score_parallel = Clifford8D.dual_stream_attention_score(v1, v2, lambda_wedge=0.5)
        self.assertGreater(score_parallel, 0.0)

        # Orthogonal vectors: alignment is zero
        v_ortho = Clifford8D.vector_8d((0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        score_ortho = Clifford8D.dual_stream_attention_score(v1, v_ortho, lambda_wedge=0.5)
        self.assertLess(score_ortho, score_parallel)


class TestCRTHodgeLaplacian(unittest.TestCase):
    def test_crt_betti_recovery_on_hypercube(self):
        c3 = DACCComplex.create_nd_hypercube(3)
        sig = TopologicalDACCInvariant.compute_signature(c3)
        # A 3-cube is contractible: beta_0 = 1, beta_1 = 0, beta_2 = 0, beta_3 = 0
        self.assertEqual(sig.beta_0, 1)
        self.assertEqual(sig.beta_1, 0)
        self.assertEqual(sig.beta_2, 0)
        self.assertEqual(sig.chi, 1)

    def test_hodge_laplacian_crt_and_roofline(self):
        c4 = DACCComplex.create_nd_hypercube(4)
        beta_1 = CombinatorialHodgeLaplacian.compute_hodge_laplacian_crt(c4, 1)
        self.assertEqual(beta_1, 0)

        roof_res = CombinatorialHodgeLaplacian.verify_roofline_machine_balance(c4, dim=1)
        self.assertIsNotNone(roof_res)
        self.assertGreater(roof_res.arithmetic_intensity, 0.0)


class TestExactProjectiveClashDetection(unittest.TestCase):
    def test_exact_rational_clash(self):
        # Two carbon atoms with overlapping van der Waals radii
        # C radius = 1.70 A. Sum = 3.40 A. Tolerance = 0.4 A => Clash if dist < 3.00 A.
        # Place at distance 1.0 A: (0, 0, 0, 1) and (1, 0, 0, 1)
        atoms_clash = [
            ("C1", "C", (0, 0, 0, 1)),
            ("C2", "C", (1, 0, 0, 1)),
        ]
        rep_clash = CGAStericClashDetector.evaluate_pairwise_clashes_rational(atoms_clash)
        self.assertFalse(rep_clash.is_clash_free)
        self.assertGreater(rep_clash.clash_count, 0)

        # Place at distance 4.0 A: (0, 0, 0, 1) and (4, 0, 0, 1)
        atoms_no_clash = [
            ("C1", "C", (0, 0, 0, 1)),
            ("C2", "C", (4, 0, 0, 1)),
        ]
        rep_free = CGAStericClashDetector.evaluate_pairwise_clashes_rational(atoms_no_clash)
        self.assertTrue(rep_free.is_clash_free)
        self.assertEqual(rep_free.clash_count, 0)


class TestPDBParserAnd8DIngestion(unittest.TestCase):
    def test_pdb_parsing_to_8d_dacc(self):
        pdb_data = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.428   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.272   2.408   0.000  1.00  0.00           O
ATOM      5  N   HIS A   2       3.332   1.545   0.000  1.00  0.00           N
ATOM      6  CA  HIS A   2       4.000   2.845   0.000  1.00  0.00           C
ATOM      7  C   HIS A   2       5.500   2.845   0.000  1.00  0.00           C
ATOM      8  O   HIS A   2       6.100   3.900   0.000  1.00  0.00           O
"""
        complex_obj = PDBParser.parse_pdb_string(pdb_data, name="Peptide_ALA_HIS", ph=6.0)
        self.assertEqual(len(complex_obj.cells[0]), 8)
        self.assertGreater(len(complex_obj.cells[1]), 0)
        # Check that 8D coordinates are attached
        for c0 in complex_obj.cells[0].values():
            coords_8d = c0.attributes.get("coordinates_8d")
            self.assertIsNotNone(coords_8d)
            self.assertEqual(len(coords_8d), 8)
            self.assertEqual(coords_8d[5], 6.0)  # pH


class TestPLQHCST8DTransformer(unittest.TestCase):
    def test_transformer_score_and_pooling(self):
        p = DACCComplex("Pocket")
        p.add_0cell("P1", coordinates_8d=(0.0, 0.0, 0.0, 0.0, 0.0, 7.4, 1.0, 0.0))
        p.add_0cell("P2", coordinates_8d=(3.0, 0.0, 0.0, 0.0, 0.0, 7.4, 1.0, 0.0))
        p.add_1cell("e_pocket", "P1", "P2")

        l = DACCComplex("Ligand")
        l.add_0cell("L1", coordinates_8d=(0.1, 0.0, 0.0, 0.0, 0.0, 6.0, 0.9, 0.0))
        l.add_0cell("L2", coordinates_8d=(2.9, 0.0, 0.0, 0.0, 0.0, 6.0, 0.9, 0.0))
        l.add_1cell("e_lig", "L1", "L2")

        transformer = PLQHCST8DTransformer()
        score_rep = transformer.score_pocket_ligand(p, l)
        self.assertEqual(score_rep.total_query_cells, 2)
        self.assertEqual(score_rep.total_key_cells, 2)
        self.assertIsInstance(score_rep.predicted_delta_g, float)

        # Test topological pooling
        pooled_p = transformer.topological_pooling_freepairs(p, dim=1)
        self.assertIsNotNone(pooled_p)


class TestDifferentiableHCSTGeneration(unittest.TestCase):
    def test_hcst_guided_generator(self):
        gen = HodgeGuidedGenerator(seed=123, max_iterations=20)
        pocket_anchors = ["Asp1", "Glu2"]
        res = gen.generate_converged_sequence_hcst(
            base_scaffold="HEHDAALLL",
            pocket_anionic_anchors=pocket_anchors,
            his_target_count=3,
        )
        self.assertIsNotNone(res.sequence)
        self.assertGreater(res.iterations, 0)
        self.assertIn("H", res.sequence)


class TestTopologicalDesolvation(unittest.TestCase):
    def test_hodge_dirac_desolvation(self):
        c = DACCComplex("DesolvTest")
        c.add_0cell("A")
        c.add_0cell("B")
        c.add_0cell("C")
        c.add_1cell("e1", "A", "B")
        c.add_1cell("e2", "B", "C")
        desolv_info = ContinuumDesolvationEngine.evaluate_topological_desolvation_hodge_dirac(
            c, screening_mass_delta_m=1.0
        )
        self.assertIn("topological_desolvation_penalty", desolv_info)
        self.assertGreaterEqual(desolv_info["topological_desolvation_penalty"], 0.0)


class TestDualTierWavefrontScheduler(unittest.TestCase):
    def test_theorem_22_wavefront_partitioning(self):
        c = DACCComplex("WavefrontComplex")
        c.add_0cell("A")
        c.add_0cell("B")
        c.add_0cell("C")
        c.add_0cell("D")

        # Two disjoint rules: A->B and C->D
        rule_specs = [
            ("rule1", ("A",), ("B",), None),
            ("rule2", ("C",), ("D",), None),
            ("rule3", ("A",), ("C",), None),  # Intersects rule1 on A and rule2 on C
        ]

        wfs = DualTierScheduler.partition_wavefronts_theorem_22(c, rule_specs)
        self.assertGreaterEqual(len(wfs), 1)
        # rule1 and rule2 are disjoint, so they can be grouped together in the first wavefront
        all_rules = [r for wf in wfs for r in wf]
        self.assertEqual(len(all_rules), 3)


if __name__ == "__main__":
    unittest.main()
