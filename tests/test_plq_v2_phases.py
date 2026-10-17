"""Comprehensive Test Suite for ToposLang & PLQ 5-Phase Upgrade."""

import unittest
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

from plq.dacc.complex import DACCComplex, PocketDACC, NilpotencyViolationError, compute_distance_filtration
from plq.dacc.fingerprint import TopologicalDACCInvariant
from plq.hodge.batched_engine import BatchedHodgeEngine
from plq.gnc.allostery import AllostericPathway, AllostericSheafSynthesizer
from plq.gnc.versor import VersorMotor
from plq.ternary.his_switch import HistidineSwitchNetwork
from plq.compiler.topos_bridge import ToposContractBridge
from plq.compiler.diff_topo import DifferentiableTopologyLayer
from plq.knots.knot_planner import KnotTopologyPlanner
from plq.sop.telemetry import ExperimentalTelemetryLoopback


class TestPhase1_VolumetricAndFiltration(unittest.TestCase):
    def test_3cell_nilpotency_success(self):
        pocket = PocketDACC("TetrahedronTest")
        cavity = pocket.add_tetrahedral_cavity("tet0", "A", "B", "C", "D", verify_nilpotency=True)
        self.assertIsNotNone(cavity)
        self.assertTrue(pocket.verify_all_nilpotence())
        sig = TopologicalDACCInvariant.compute_signature(pocket)
        self.assertEqual(sig.chi, 1)
        self.assertEqual(sig.num_3cells, 1)
        self.assertTrue(sig.is_druggable_3d)

    def test_3cell_nilpotency_failure(self):
        c = DACCComplex("Invalid3Cell")
        # Build triangle face f0
        c.add_0cell("v0")
        c.add_0cell("v1")
        c.add_0cell("v2")
        c.add_1cell("e01", "v0", "v1")
        c.add_1cell("e12", "v1", "v2")
        c.add_1cell("e02", "v0", "v2")
        c.add_2cell("f0", [("e01", 1), ("e12", 1), ("e02", -1)])
        # Single face boundary does not form a closed 2-cycle => non-zero boundary!
        with self.assertRaises(NilpotencyViolationError):
            c.add_3cell("c3_bad", [("f0", 1)], verify_nilpotency=True)

    def test_distance_filtration(self):
        coords = {
            "v1": (0.0, 0.0, 0.0),
            "v2": (3.0, 0.0, 0.0),
            "v3": (1.5, 2.5, 0.0),
        }
        res = compute_distance_filtration(coords, cutoffs=[2.0, 3.5, 4.0])
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0]["beta_0"], 3)  # no edges at r=2.0
        self.assertLess(res[1]["beta_0"], 3)   # edges connect at r=3.5


class TestPhase2_BatchedHodgeEngine(unittest.TestCase):
    def test_batched_screening(self):
        engine = BatchedHodgeEngine(batch_size=32)
        candidates = [
            ("cand_1", "HEHDAALLL"),
            ("cand_2", "QVQLVESGGGLVQPGGSLRLSCAASGFT"),
            ("cand_3", "MALWMRLLPLLALLALWGPDPAAAFVN"),
        ]
        res = engine.screen_library_batched(candidates)
        self.assertEqual(res.total_candidates, 3)
        self.assertGreater(res.throughput_per_second, 0.0)


class TestPhase3_AllosteryAndMicroenvironment(unittest.TestCase):
    def test_allosteric_sheaf_synthesis(self):
        c = DACCComplex("AllosteryTest")
        for n in ["A", "B", "C"]:
            c.add_0cell(n)
        pathway = AllostericPathway(
            name="TestWire",
            nodes=["A", "B", "C"],
            input_motor=VersorMotor.identity(),
            target_output_motor=VersorMotor(tx=1.0, ty=0.0, tz=0.0, rz=0.1),
        )
        res = AllostericSheafSynthesizer.synthesize_allosteric_sheaf(c, pathway)
        self.assertEqual(res.obstruction_dim_h1, 0)
        self.assertTrue(res.is_frustration_free)

    def test_microenvironment_pka_shift(self):
        switch = HistidineSwitchNetwork()
        shifted = switch.compute_microenvironment_pka(local_anionic_contacts=3, mean_contact_distance=3.5)
        self.assertGreater(shifted, switch.pka)
        # Shifted pKa should produce higher protonation fraction at pH 7.4
        f_base = switch.protonation_fraction(7.4)
        f_shifted = switch.protonation_fraction(7.4, shifted)
        self.assertGreater(f_shifted, f_base)


class TestPhase4_ToposBridgeAndDiffTopo(unittest.TestCase):
    def test_topos_contract_compilation(self):
        contract_src = """
        contract TestContract {
            cell 0 a;
            cell 0 b;
            cell 0 c;
            cell 1 e1 : a -> b;
            cell 1 e2 : b -> c;
            cell 1 e3 : a -> c;
            cell 2 f1 { e1: 1, e2: 1, e3: -1 };
            assert nilpotency == true;
            assert betti(1) == 0;
            assert euler_characteristic == 1;
        }
        """
        complex_obj, report = ToposContractBridge.compile_contract(contract_src)
        self.assertTrue(report.nilpotency_verified)
        self.assertEqual(report.beta_1, 0)
        self.assertTrue(report.all_assertions_passed)

    def test_diff_topo_loss(self):
        diff_layer = DifferentiableTopologyLayer()
        adj = [
            [0.0, 0.8, 0.8],
            [0.8, 0.0, 0.8],
            [0.8, 0.8, 0.0],
        ]
        res, grad = diff_layer.compute_loss_and_gradients(adj)
        self.assertIsInstance(res.loss, float)
        self.assertEqual(len(grad), 3)


class TestPhase5_KnotsAndSOPTelemetry(unittest.TestCase):
    def test_knot_classification(self):
        crossings = [
            ("a1", "a2", "a3", 1),
            ("a2", "a3", "a1", 1),
            ("a3", "a1", "a2", 1),
        ]
        res = KnotTopologyPlanner.evaluate_knot_topology(crossings)
        self.assertTrue(res.is_mechanically_knotted)
        self.assertIn("Trefoil", res.knot_identifier)

    def test_sop_telemetry_loopback(self):
        # SOP-1: N_His >= 4 with ratio >= 100
        sop1 = ExperimentalTelemetryLoopback.evaluate_sop1_spr("C1", kd_74_nm=1.0, kd_60_nm=120.0, his_count=4)
        self.assertTrue(sop1.t4_switch_confirmed)

        # SOP-2: Helical minibinder with Tm >= 60 and f_helix >= 0.35
        sop2 = ExperimentalTelemetryLoopback.evaluate_sop2_cd("C1", "de_novo_minibinder", f_helix=0.45, f_sheet=0.1, tm_celsius=68.0)
        self.assertTrue(sop2.acceptance_passed)

        # SOP-3: Delta Tm <= -10 for N_His >= 4
        sop3 = ExperimentalTelemetryLoopback.evaluate_sop3_dsf("C1", tm_74=70.0, tm_60=55.0, his_count=4)
        self.assertTrue(sop3.percolation_destabilization_confirmed)

        # SOP-4: Matching Betti numbers
        coords = {"r1": (0, 0, 0), "r2": (3, 0, 0)}
        sop4 = ExperimentalTelemetryLoopback.evaluate_sop4_cryoem("C1", 2.8, predicted_beta1=0, interface_coords=coords)
        self.assertTrue(sop4.topological_match)


if __name__ == "__main__":
    unittest.main()
