"""Comprehensive Unit and Integration Test Suite for PLQ Core Engine."""

import unittest
from fractions import Fraction
import sys
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

from plq.dacc.complex import DACCComplex, MolecularDACC, PocketDACC, NilpotencyViolationError
from plq.dacc.fingerprint import TopologicalDACCInvariant
from plq.gnc.clifford import Multivector, PGA3D, CGA3D
from plq.gnc.versor import VersorMotor, CoordinateFreeAligner
from plq.gnc.sheaf import CliffordSheaf
from plq.ternary.lattice import TernaryLattice, TernaryState
from plq.ternary.his_switch import HistidineSwitchNetwork, BifoldManifold
from plq.polypharmacology.dual_tier import DualTierScheduler, TargetScreeningTask
from plq.hodge.laplacian import CombinatorialHodgeLaplacian
from plq.hodge.engine import HodgeGuidedGenerator
from plq.compiler.symbolic_gnc import SymbolicGNCPass


class TestDACCComplex(unittest.TestCase):
    def test_boundary_nilpotency_success(self):
        c = DACCComplex("Triangle")
        c.add_0cell("v0")
        c.add_0cell("v1")
        c.add_0cell("v2")
        c.add_1cell("e01", "v0", "v1")
        c.add_1cell("e12", "v1", "v2")
        c.add_1cell("e02", "v0", "v2")
        # d_1(e01) = v1 - v0, d_1(e12) = v2 - v1, d_1(e02) = v2 - v0
        # Cycle: +e01 + e12 - e02 -> (v1-v0) + (v2-v1) - (v2-v0) = 0
        face = c.add_2cell("f0", [("e01", 1), ("e12", 1), ("e02", -1)], verify_nilpotency=True)
        self.assertIsNotNone(face)
        self.assertTrue(c.verify_all_nilpotence())

    def test_boundary_nilpotency_failure(self):
        c = DACCComplex("Invalid")
        c.add_0cell("v0")
        c.add_0cell("v1")
        c.add_0cell("v2")
        c.add_1cell("e01", "v0", "v1")
        c.add_1cell("e12", "v1", "v2")
        # Invalid boundary (not a closed cycle): +e01 + e12 -> v2 - v0 != 0
        with self.assertRaises(NilpotencyViolationError):
            c.add_2cell("f_bad", [("e01", 1), ("e12", 1)], verify_nilpotency=True)

    def test_molecular_dacc_from_sequence(self):
        mol = MolecularDACC.from_sequence("Peptide_Test", "HEHDA")
        self.assertEqual(len(mol.cells[0]), 5)
        self.assertTrue(mol.verify_all_nilpotence())
        sig = TopologicalDACCInvariant.compute_signature(mol)
        self.assertEqual(sig.beta_0, 1)
        self.assertGreater(sig.his_switch_density, 0.0)


class TestCliffordGNC(unittest.TestCase):
    def test_pga_sandwich_identity(self):
        p = PGA3D.point(1.0, 2.0, 3.0)
        motor = VersorMotor.identity()
        transformed = motor.apply(p)
        self.assertAlmostEqual(transformed.norm(), p.norm(), places=5)

    def test_cga_point_and_sphere_wedge(self):
        # Two identical spheres should have zero wedge product (coincident)
        s1 = CGA3D.sphere(0.0, 0.0, 0.0, 2.0)
        s2 = CGA3D.sphere(0.0, 0.0, 0.0, 2.0)
        overlap = CGA3D.test_intersection(s1, s2)
        self.assertAlmostEqual(overlap, 0.0, places=5)

    def test_coordinate_free_aligner(self):
        pocket = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        binder = [(2.0, 3.0, 4.0), (3.0, 3.0, 4.0), (2.0, 4.0, 4.0)]
        motor, rmsd = CoordinateFreeAligner.align_anchors(pocket, binder)
        self.assertAlmostEqual(rmsd, 0.0, places=3)


class TestTernaryLatticeAndHistidineSwitch(unittest.TestCase):
    def test_his_switch_bifold(self):
        switch = HistidineSwitchNetwork(pka=6.2)
        # Sequence with histidines: "HEHDH"
        eval_res = switch.evaluate_bifold_manifold("HEHDHAALLL", pocket_anionic_count=3, pocket_hydrophobic_count=4)
        # pH 6.5 should bind much stronger than pH 7.4
        self.assertLess(eval_res.delta_g_65, eval_res.delta_g_74)
        self.assertGreater(eval_res.delta_delta_g, 2.0)
        self.assertTrue(eval_res.is_switch_active)


class TestHodgeLaplacian(unittest.TestCase):
    def test_hodge_laplacian_betti(self):
        # Triangle with closed face: beta_1 should be 0
        c = DACCComplex("Triangle")
        c.add_0cell("v0")
        c.add_0cell("v1")
        c.add_0cell("v2")
        c.add_1cell("e01", "v0", "v1")
        c.add_1cell("e12", "v1", "v2")
        c.add_1cell("e02", "v0", "v2")
        c.add_2cell("f0", [("e01", 1), ("e12", 1), ("e02", -1)])
        l1, beta_1 = CombinatorialHodgeLaplacian.compute_hodge_laplacian_1(c)
        self.assertEqual(beta_1, 0)

    def test_hodge_guided_generation(self):
        generator = HodgeGuidedGenerator(seed=123, max_iterations=20)
        res = generator.generate_converged_sequence(
            base_scaffold="QVQLVESGGGLVQPGGSLRLSCAASGFT",
            pocket_anionic_anchors=["Asp355", "Glu384", "Gln408"],
            his_target_count=3,
        )
        self.assertEqual(res.final_beta_1, 0)
        self.assertTrue(res.converged)


class TestDualTierAndCompiler(unittest.TestCase):
    def test_dual_tier_scheduling(self):
        tasks = [
            TargetScreeningTask("t1", "EGFR_Human", "B1", discrete_support={"pocket_A", "pocket_B"}, evaluate_fn=lambda: "done1"),
            TargetScreeningTask("t2", "EGFR_Mouse", "B2", discrete_support={"pocket_C", "pocket_D"}, evaluate_fn=lambda: "done2"),
            TargetScreeningTask("t3", "HER2", "B3", discrete_support={"pocket_A", "pocket_E"}, evaluate_fn=lambda: "done3"),
        ]
        wavefronts = DualTierScheduler.partition_discrete_wavefronts(tasks)
        # t1 and t2 have disjoint support, so they should be in the same wavefront
        self.assertEqual(len(wavefronts[0].tasks), 2)
        results = DualTierScheduler.execute_screening_parallel(tasks, max_workers=2)
        self.assertEqual(len(results), 3)

    def test_symbolic_gnc_compiler_pass(self):
        mol = MolecularDACC.from_sequence("TestMol", "HAL")
        p1 = CGA3D.point(1.0, 2.0, 3.0)
        compiled = SymbolicGNCPass.compile(mol, sample_multivectors=[p1])
        self.assertIn("topos.gnc_version", compiled.mlir_ir)
        self.assertGreater(len(compiled.simd_cache_lines), 0)


if __name__ == "__main__":
    unittest.main()
