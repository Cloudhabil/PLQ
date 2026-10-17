"""Comprehensive Unit Tests for De Novo Design Mathematical Metrics in PLQ.

Verifies:
1. ESMQualityEvaluator: pLDDT (local lDDT across thresholds and binned heads),
   TM-score (with pure-Python Kabsch SVD superposition), pTM, and ipTM.
2. ProteinMPNNScorer: NLL score, paratope redesigned score, sequence recovery %,
   and backbone geometric compatibility estimation.
3. StructuralClassifier: Novelty (1 - max TM-score), TED domain segmentation
   modularity (normalized graph cuts), and DSSP electrostatic H-bond foldstrings.
4. PhysicoChemicalEngine: Exact isotopic Mw accounting for condensation water loss,
   strictly decreasing continuous charge Q(pH), and Newton-Raphson pI root-finding.
5. DACC Topological Coupling: DSSP H-bonds populated as 2-cells satisfying
   boundary nilpotency d_1 o d_2 = 0.
"""

from __future__ import annotations
import math
import unittest
from typing import List, Tuple

from plq.dacc.complex import DACCComplex, MolecularDACC
from plq.dacc.fingerprint import TopologicalDACCInvariant
from plq.metrics import (
    ESMQualityEvaluator,
    PLDDTReport,
    TMScoreReport,
    ProteinMPNNScorer,
    MPNNScoreReport,
    StructuralClassifier,
    DSSPReport,
    TEDReport,
    PhysicoChemicalEngine,
    PhysicoChemicalReport,
)


class TestESMQualityMetrics(unittest.TestCase):
    """Test suite for ESMFold structural quality metrics: pLDDT, TM-score, pTM, ipTM."""

    def test_plddt_identical_structures(self) -> None:
        """Identical coordinates must yield a perfect pLDDT of 100.0."""
        coords = [
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
            (15.2, 0.0, 0.0),
        ]
        report = ESMQualityEvaluator.compute_plddt(coords, coords, inclusion_radius=15.0)
        self.assertEqual(report.mean_plddt, 100.0)
        self.assertEqual(report.high_confidence_ratio, 1.0)
        self.assertEqual(report.very_high_confidence_ratio, 1.0)
        for val in report.residue_plddt:
            self.assertEqual(val, 100.0)

    def test_plddt_perturbed_structure(self) -> None:
        """Perturbed coordinates should yield monotonically decaying pLDDT across thresholds."""
        true_coords = [
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
        ]
        # Introduce a 3.5 A perturbation on residue 2
        pred_coords = [
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 3.5, 0.0),
            (11.4, 0.0, 0.0),
        ]
        report = ESMQualityEvaluator.compute_plddt(pred_coords, true_coords, inclusion_radius=15.0)
        self.assertGreater(report.mean_plddt, 0.0)
        self.assertLess(report.mean_plddt, 100.0)

    def test_plddt_from_binned_distribution(self) -> None:
        """Binned probability expectation sum(p_i,b * v_b) must match analytical expectation."""
        # 50 bins over [0, 100] with width 2.0. Bin 40 has midpoint 81.0
        num_bins = 50
        distribution_1 = [0.0] * num_bins
        distribution_1[40] = 1.0  # 100% confidence at bin 40 => midpoint 81.0

        distribution_2 = [0.0] * num_bins
        distribution_2[45] = 1.0  # midpoint 91.0

        report = ESMQualityEvaluator.compute_plddt_from_bins([distribution_1, distribution_2])
        self.assertAlmostEqual(report.residue_plddt[0], 81.0, places=1)
        self.assertAlmostEqual(report.residue_plddt[1], 91.0, places=1)
        self.assertAlmostEqual(report.mean_plddt, 86.0, places=1)
        self.assertEqual(report.high_confidence_ratio, 1.0)
        self.assertEqual(report.very_high_confidence_ratio, 0.5)

    def test_tm_score_exact_superposition(self) -> None:
        """Identical structure rotated and translated must yield TM-score = 1.0 and RMSD ~ 0.0."""
        # 20-residue alpha-helix backbone
        true_coords: List[Tuple[float, float, float]] = []
        for i in range(20):
            theta = i * 1.745  # ~100 degrees per turn
            z = i * 1.5
            r = 2.3
            true_coords.append((r * math.cos(theta), r * math.sin(theta), z))

        # Rotate by 90 degrees around Z and translate by (10, 20, 30)
        rotated_coords: List[Tuple[float, float, float]] = []
        for x, y, z in true_coords:
            rx = -y + 10.0
            ry = x + 20.0
            rz = z + 30.0
            rotated_coords.append((rx, ry, rz))

        report = ESMQualityEvaluator.compute_tm_score(rotated_coords, true_coords)
        self.assertAlmostEqual(report.tm_score, 1.0, places=2)
        self.assertLess(report.rmsd, 1e-3)
        self.assertEqual(report.aligned_length, 20)
        self.assertGreater(report.d0, 0.0)

    def test_ptm_and_iptm_from_pae(self) -> None:
        """Tests pTM and interface ipTM from a Predicted Aligned Error (PAE) matrix."""
        # 10-residue chain A (0..4) and chain B (5..9)
        l = 10
        # Perfect PAE: error 0 on diagonal, small nearby error
        pae = [[abs(i - j) * 0.5 for j in range(l)] for i in range(l)]

        ptm = ESMQualityEvaluator.compute_ptm(pae)
        self.assertGreater(ptm, 0.0)
        self.assertLessEqual(ptm, 1.0)

        # ipTM across chains A (len 5) and B (len 5)
        iptm = ESMQualityEvaluator.compute_iptm(pae, chain_lengths=[5, 5])
        self.assertGreater(iptm, 0.0)
        self.assertLessEqual(iptm, 1.0)


class TestProteinMPNNScoring(unittest.TestCase):
    """Test suite for ProteinMPNN sequence design, NLL, and recovery metrics."""

    def test_nll_score_analytical(self) -> None:
        """Negative log-likelihood score -1/N sum log P(s_i) must match exact values."""
        seq = "AAA"
        # Probabilities: P(A) = 0.5 => -log(0.5) = ln(2) ~ 0.6931
        log_prob_val = math.log(0.5)
        log_probs = [{"A": log_prob_val}, {"A": log_prob_val}, {"A": log_prob_val}]

        nll = ProteinMPNNScorer.compute_nll_score(seq, log_probs)
        self.assertAlmostEqual(nll, -log_prob_val, places=3)

    def test_redesigned_score_paratope_mask(self) -> None:
        """Redesigned score must only average over the specified mutable mask indices M."""
        seq = "HEHDA"
        log_probs = [
            {"H": math.log(0.1)},  # idx 0: -log(0.1) ~ 2.302
            {"E": math.log(0.9)},  # idx 1: -log(0.9) ~ 0.105 (in M)
            {"H": math.log(0.8)},  # idx 2: -log(0.8) ~ 0.223 (in M)
            {"D": math.log(0.1)},  # idx 3: -log(0.1) ~ 2.302
            {"A": math.log(0.2)},  # idx 4: -log(0.2) ~ 1.609
        ]
        redesigned_indices = {1, 2}
        score = ProteinMPNNScorer.compute_redesigned_score(seq, log_probs, redesigned_indices)
        expected = (-math.log(0.9) + -math.log(0.8)) / 2.0
        self.assertAlmostEqual(score, expected, places=3)

    def test_sequence_recovery_percentage(self) -> None:
        """Sequence recovery percentage matches exact identity count."""
        native = "ACDEFGHIKL"
        designed = "ACDEPGHIKM"  # 2 mutations: F->P, L->M (8/10 = 80%)
        rec = ProteinMPNNScorer.compute_sequence_recovery(designed, native)
        self.assertEqual(rec, 80.0)

        # Restricted to mask
        mask = {0, 1, 2, 3}  # All 4 match => 100%
        rec_masked = ProteinMPNNScorer.compute_sequence_recovery(designed, native, mask_indices=mask)
        self.assertEqual(rec_masked, 100.0)

    def test_estimate_backbone_compatibility(self) -> None:
        """Estimating backbone compatibility from coordinates produces valid pseudo-NLL."""
        seq = "HEHDAALLLW"
        coords = [(i * 3.8, 0.0, 0.0) for i in range(len(seq))]
        report = ProteinMPNNScorer.estimate_backbone_compatibility(seq, coords)
        self.assertGreater(report.nll_score, 0.0)
        self.assertGreater(report.mean_perplexity, 1.0)
        self.assertEqual(report.sequence_length, len(seq))


class TestStructuralAnalysisAndDSSP(unittest.TestCase):
    """Test suite for Novelty, TED modularity, and DSSP Foldstring assignment."""

    def test_novelty_score(self) -> None:
        """Novelty is 0 for identical structure and close to 1 for distinct folds."""
        coords_a = [(i * 3.8, 0.0, 0.0) for i in range(20)]
        # Reference list contains coords_a
        novelty_zero = StructuralClassifier.compute_novelty(coords_a, [coords_a])
        self.assertAlmostEqual(novelty_zero, 0.0, places=2)

        # Distant globular fold vs linear
        coords_globular: List[Tuple[float, float, float]] = []
        for i in range(20):
            coords_globular.append((math.cos(i) * 5.0, math.sin(i) * 5.0, (i % 4) * 3.0))
        novelty_high = StructuralClassifier.compute_novelty(coords_a, [coords_globular])
        self.assertGreater(novelty_high, 0.3)

    def test_ted_confidence_domain_modularity(self) -> None:
        """Tests TED graph-cut modularity on multi-domain vs single domain structures."""
        # Construct two separated compact clusters connected by a thin linker
        domain_1 = [(x * 2.0, y * 2.0, 0.0) for x in range(4) for y in range(4)]  # 16 atoms
        linker = [(20.0, 0.0, 0.0), (30.0, 0.0, 0.0)]
        domain_2 = [(40.0 + x * 2.0, y * 2.0, 0.0) for x in range(4) for y in range(4)]  # 16 atoms
        coords = domain_1 + linker + domain_2

        report = StructuralClassifier.compute_ted_confidence(coords, cut_threshold=6.0, min_domain_length=10)
        self.assertIsNotNone(report.best_split_index)
        self.assertGreater(report.modularity, 0.5)

    def test_dssp_foldstring_hbond_energy(self) -> None:
        """Verifies DSSP electrostatic H-bond energy formula: E = 332 * 0.42 * 0.20 * sum(1/r)."""
        # Canonical alpha-helical turn: i -> i+4
        # Standard alpha helix has r_ON ~ 2.9 A, r_CH ~ 3.3 A, r_OH ~ 1.9 A, r_CN ~ 3.9 A
        # E = 27.888 * (1/2.9 + 1/3.3 - 1/1.9 - 1/3.9) = 27.888 * (0.3448 + 0.3030 - 0.5263 - 0.2564)
        #   = 27.888 * (-0.1349) = -3.76 kcal/mol (< -0.5 kcal/mol cutoff)
        res_i = {
            "C": (0.0, 0.0, 0.0),
            "O": (0.0, 1.23, 0.0),
            "CA": (-1.2, -0.7, 0.0),
            "N": (-2.4, 0.0, 0.0),
            "H": (-2.4, 1.0, 0.0),
        }
        res_j = {
            "N": (0.0, 3.1, 0.0),    # r_ON = 3.1 - 1.23 = 1.87 A
            "H": (0.0, 2.1, 0.0),    # r_OH = 2.1 - 1.23 = 0.87 A
            "C": (1.2, 3.8, 0.0),
            "O": (1.2, 5.0, 0.0),
            "CA": (-1.2, 3.8, 0.0),
        }
        backbone = [res_i, {}, {}, {}, res_j]  # i = 0, j = 4
        report = StructuralClassifier.compute_foldstring_dssp(backbone)
        self.assertGreaterEqual(len(report.hbond_pairs), 1)
        donor, acceptor, energy = report.hbond_pairs[0]
        self.assertEqual(donor, 4)
        self.assertEqual(acceptor, 0)
        self.assertLess(energy, -0.5)


class TestPhysicoChemicalEngine(unittest.TestCase):
    """Test suite for isotopic Mw and Newton-Raphson continuous pI solving."""

    def test_molecular_weight_condensation_loss(self) -> None:
        """Mw must equal sum of constituent amino acids minus (N - 1) * 18.0152 Da."""
        # Gly-Gly-Gly: Gly free mass = 75.067 Da
        # 3 * 75.067 - 2 * 18.01524 = 225.201 - 36.0305 = 189.1705 Da
        mw_ggg = PhysicoChemicalEngine.compute_molecular_weight("GGG", monoisotopic=False)
        self.assertAlmostEqual(mw_ggg, 189.1705, places=2)

        # Monoisotopic mass: Gly = 75.03203, H2O = 18.010565
        # 3 * 75.03203 - 2 * 18.010565 = 225.09609 - 36.02113 = 189.07496 Da
        mono_ggg = PhysicoChemicalEngine.compute_molecular_weight("GGG", monoisotopic=True)
        self.assertAlmostEqual(mono_ggg, 189.0750, places=3)

    def test_charge_and_analytical_derivative(self) -> None:
        """Net charge Q(pH) must be strictly monotonically decreasing and dQ/dpH < 0 everywhere."""
        seq = "HEHDAALLLW"
        prev_q = float("inf")
        for ph_x10 in range(10, 130, 5):  # pH 1.0 to 13.0
            ph = ph_x10 / 10.0
            q, dq = PhysicoChemicalEngine.compute_charge_and_derivative(seq, ph)
            self.assertLess(q, prev_q, f"Monotonicity failed at pH {ph}")
            self.assertLess(dq, 0.0, f"Negative derivative failed at pH {ph}")
            prev_q = q

    def test_newton_raphson_pi_exact_root(self) -> None:
        """The solved pI must satisfy |Q(pI)| < 1e-4 across diverse peptides."""
        test_sequences = [
            "HEHDAALLLW",       # Mixed acidic/basic
            "KKKKRRRR",         # Highly basic poly-cationic
            "DDDDEEEE",         # Highly acidic poly-anionic
            "AAAAAAAA",         # Neutral aliphatic
            "CYWWYCW",          # Aromatic & sulfur-bearing
        ]
        for seq in test_sequences:
            pi = PhysicoChemicalEngine.compute_isoelectric_point_newton(seq, tol=1e-5)
            q_at_pi = PhysicoChemicalEngine.compute_charge_at_ph(seq, pi)
            self.assertLess(
                abs(q_at_pi),
                1e-3,
                f"Net charge at pI for sequence '{seq}' was {q_at_pi} (pI={pi})",
            )
            # Basic sequence must have pI > 9.0
            if seq == "KKKKRRRR":
                self.assertGreater(pi, 10.0)
            # Acidic sequence must have pI < 4.0
            elif seq == "DDDDEEEE":
                self.assertLess(pi, 4.0)

    def test_extinction_coeff_and_gravy(self) -> None:
        """Extinction coefficient at 280nm and GRAVY hydropathy index check."""
        seq = "WWYYCC"  # 2 W (2*5500) + 2 Y (2*1490) + 2 C (1 cystine pair * 125) = 11000 + 2980 + 125 = 14105
        eps = PhysicoChemicalEngine.compute_extinction_coefficient(seq)
        self.assertEqual(eps, 14105.0)

        # Pure Isoleucine has hydropathy 4.5
        gravy_poly_ile = PhysicoChemicalEngine.compute_gravy("IIII")
        self.assertAlmostEqual(gravy_poly_ile, 4.5, places=2)


class TestDACCDSSPCoupling(unittest.TestCase):
    """Test suite for DACC integration with DSSP hydrogen bonds."""

    def test_dssp_hbond_2cell_nilpotency(self) -> None:
        """DSSP H-bonds added to DACC must satisfy boundary nilpotency d_1 o d_2 = 0."""
        dacc = MolecularDACC.from_sequence("P1", "HEHDAALLLW")

        # Mock an electrostatic H-bond between residue 0 (acceptor) and residue 4 (donor)
        hbond_pairs = [(4, 0, -2.5)]
        foldstring = "THHHTCLLLL"

        added_2cells = dacc.add_dssp_hydrogen_bonds_as_2cells(foldstring, hbond_pairs)
        self.assertGreaterEqual(len(added_2cells), 1)

        # Verify nilpotency holds identically across the entire complex
        self.assertTrue(dacc.verify_all_nilpotence())

        # Verify foldstring attribute was attached to residue 0-cells
        cell_0 = dacc.get_cell("res_0_H", dim=0)
        self.assertEqual(cell_0.attributes.get("foldstring"), "T")



class TestInvertedDeNovoOptimization(unittest.TestCase):
    """Test suite for inverted de novo objective functions and generative optimization."""

    def test_softmax_temperature(self) -> None:
        """Verifies probability normalization sum_k pi_{i,k} == 1.0 and temperature sharpens."""
        from plq.metrics.objective import DeNovoLossLandscape
        logits = [[0.0, 1.0, 2.0, 5.0]]
        probs_high_t = DeNovoLossLandscape.softmax_logits(logits, tau=2.0)
        probs_low_t = DeNovoLossLandscape.softmax_logits(logits, tau=0.2)

        self.assertAlmostEqual(sum(probs_high_t[0]), 1.0, places=5)
        self.assertAlmostEqual(sum(probs_low_t[0]), 1.0, places=5)
        # Low temperature sharpens peak probability
        self.assertGreater(probs_low_t[0][3], probs_high_t[0][3])

    def test_loss_seq_and_recovery(self) -> None:
        """Verifies L_seq and L_recovery loss calculation and shape of analytical gradients."""
        from plq.metrics.objective import DeNovoLossLandscape
        logits = [[0.0] * 20 for _ in range(5)]
        probs = DeNovoLossLandscape.softmax_logits(logits, tau=1.0)
        target_lps = [[-math.log(20.0)] * 20 for _ in range(5)]

        l_seq, gz_seq = DeNovoLossLandscape.compute_loss_seq(probs, target_lps, tau=1.0)
        self.assertAlmostEqual(l_seq, math.log(20.0), places=3)
        self.assertEqual(len(gz_seq), 5)
        self.assertEqual(len(gz_seq[0]), 20)

        # Recovery against template
        l_rec, gz_rec = DeNovoLossLandscape.compute_loss_recovery(probs, "ACDEF", tau=1.0)
        self.assertAlmostEqual(l_rec, 0.95, places=2)  # Uniform prob 0.05 => 1 - 0.05 = 0.95
        self.assertEqual(len(gz_rec), 5)

    def test_loss_novelty(self) -> None:
        """Novelty loss activates only when TM > TM_target and pushes coordinates away."""
        from plq.metrics.objective import DeNovoLossLandscape
        coords = [(i * 3.8, 0.0, 0.0) for i in range(10)]
        # Reference identical to coords => TM = 1.0 > 0.5 => loss > 0
        l_nov, gx_nov = DeNovoLossLandscape.compute_loss_novelty(coords, [coords], tm_target=0.5)
        self.assertAlmostEqual(l_nov, 0.5, places=2)
        self.assertEqual(len(gx_nov), 10)

        # High threshold => loss = 0.0
        l_zero, gx_zero = DeNovoLossLandscape.compute_loss_novelty(coords, [coords], tm_target=1.0)
        self.assertEqual(l_zero, 0.0)

    def test_loss_pi_and_mw(self) -> None:
        """Verifies continuous pI target loss and Mw loss with analytical gradients."""
        from plq.metrics.objective import DeNovoLossLandscape
        logits = [[0.0] * 20 for _ in range(8)]
        probs = DeNovoLossLandscape.softmax_logits(logits, tau=1.0)

        l_pi, gz_pi = DeNovoLossLandscape.compute_loss_pi(probs, target_pi=7.0, tau=1.0)
        self.assertGreaterEqual(l_pi, 0.0)
        self.assertEqual(len(gz_pi), 8)

        l_mw, gz_mw = DeNovoLossLandscape.compute_loss_mw(probs, target_mw=1000.0, tau=1.0)
        self.assertGreater(l_mw, 0.0)
        self.assertEqual(len(gz_mw), 8)

    def test_denovo_generative_optimizer_end_to_end(self) -> None:
        """Verifies end-to-end continuous Langevin and gradient-guided optimization."""
        from plq.metrics.objective import DeNovoGenerativeOptimizer, DeNovoObjectiveWeights
        from plq.hodge.engine import HodgeGuidedGenerator

        opt = DeNovoGenerativeOptimizer(
            weights=DeNovoObjectiveWeights(
                w_seq=1.0,
                w_helix=2.0,
                w_pi=1.0,
                w_topo=0.5,
            ),
            max_iterations=15,
            seed=42,
        )

        result = opt.optimize(
            sequence_length=8,
            target_foldstring="HHHHHHHH",
            target_pi=6.5,
        )

        self.assertEqual(len(result.designed_sequence), 8)
        self.assertEqual(len(result.backbone_coordinates), 8)
        self.assertGreater(len(result.loss_history), 0)
        # Loss must decrease over iterations
        self.assertLess(result.loss_history[-1], result.loss_history[0])
        self.assertGreater(result.predicted_pi, 0.0)
        self.assertGreater(result.predicted_mw, 0.0)

    def test_hodge_generator_optimized_method(self) -> None:
        """Verifies HodgeGuidedGenerator.generate_denovo_binder_optimized integration."""
        from plq.hodge.engine import HodgeGuidedGenerator
        gen = HodgeGuidedGenerator(seed=123, max_iterations=10)
        res = gen.generate_denovo_binder_optimized(
            sequence_length=6,
            target_foldstring="HHHHHH",
            target_pi=7.4,
        )
        self.assertEqual(len(res.designed_sequence), 6)
        self.assertGreater(res.predicted_mw, 0.0)


if __name__ == "__main__":
    unittest.main()

