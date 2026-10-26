"""Unit & Integration tests for PLQ Wave Sheaf Solvation, Bifold Cascade, and HCST/Langevin coupling."""

import unittest

from plq.dacc.complex import DACCComplex
from plq.gnc.allostery import SynergeiaBifoldaCascade
from plq.gnc.sheaf import AnisotropiaDielectricaSheaf
from plq.metrics.dispersive_attention import DispersiveHodgeCliffordAttention
from plq.metrics.objective import DeNovoGenerativeOptimizer, DeNovoObjectiveWeights
from plq.metrics.solvation_entropy import DynamicSolvationEntropy
from plq.neural.hcst_pocket import PLQHCST8DTransformer
from plq.physics.wave_pocket import PocketWaveField


class TestPLQWaveSheafIntegration(unittest.TestCase):
    """Verifies calibrated dielectric sheaf, Campbell entropy, bifold cascade, and HCST/Langevin integration."""

    def test_anisotropia_dielectrica_and_campbell_agreement(self) -> None:
        sheaf = AnisotropiaDielectricaSheaf()
        solv = DynamicSolvationEntropy()
        r_iface = [3.0, 4.2, 5.5, 7.0]
        s_sheaf = sheaf.compute_dynamic_solvation_entropy(r_iface, t=1.0, c=1750.0)
        s_metric = solv.compute(r_iface, t=1.0).t_ds_solv_total
        self.assertAlmostEqual(s_sheaf, s_metric, places=5)
        self.assertGreater(s_sheaf, 0.1)
        self.assertLess(s_sheaf, 5.0)

    def test_hydrophobic_attractor_and_mcmc(self) -> None:
        solv = DynamicSolvationEntropy()
        phi_star = solv.find_hydrophobic_attractor(n_steps=300)
        self.assertAlmostEqual(phi_star, 0.2788, places=3)
        mean_phi, std_phi, collapse_rate = solv.mcmc_hydrophobic_sampling(
            n_steps=20_000, step_size=0.05, kbt_effective=0.048, seed=42
        )
        self.assertAlmostEqual(mean_phi, 0.279, delta=0.015)
        self.assertEqual(collapse_rate, 0.0)

    def test_synergeia_bifolda_heaviside_switch(self) -> None:
        cascade = SynergeiaBifoldaCascade()
        tel = cascade.evaluate_switch_telemetry()
        self.assertGreater(tel.delta_q_surge, 2.80)
        self.assertAlmostEqual(tel.hill_coefficient, 2.91, delta=0.12)
        self.assertTrue(tel.is_heaviside_switch)

    def test_dispersive_attention_and_pocket_wave(self) -> None:
        attn = DispersiveHodgeCliffordAttention(lam=0.5, use_dispersive_envelope=True)
        q = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        k = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        res = attn.compute_attention_matrix(q, k, [(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)], t=0.5)
        self.assertEqual(res.n_queries, 1)
        self.assertEqual(res.n_keys, 1)

        pocket = PocketWaveField([3.0, 5.0, 7.0], c_wave=1750.0, shell_thickness=1.0)
        pocket.inject_ligand_approach(r_source=0.0, t_impact=0.0, amplitude=1.0)
        pr = pocket.propagate(t=3.0 / 1750.0)
        self.assertGreater(pr.kon_proxy, 0.0)

    def test_hcst_8d_dispersive_pocket_scoring(self) -> None:
        pocket = DACCComplex("TargetPocket")
        pocket.add_0cell("p0", coordinates_8d=(0.0, 0.0, 0.0, -60.0, -45.0, 7.4, -1.0, 0.2))
        pocket.add_0cell("p1", coordinates_8d=(3.8, 0.0, 0.0, -60.0, -45.0, 7.4, -1.0, 0.2))

        ligand = DACCComplex("CandidateBinder")
        ligand.add_0cell("l0", coordinates_8d=(3.5, 0.0, 0.0, -60.0, -45.0, 6.0, 1.0, 0.2))
        ligand.add_0cell("l1", coordinates_8d=(7.0, 0.0, 0.0, -60.0, -45.0, 6.0, 1.0, 0.2))

        hcst = PLQHCST8DTransformer()
        profile = hcst.score_pocket_ligand_dispersive(pocket, ligand, time_ps=1.0)
        self.assertIn("predicted_kd_nm", profile)
        self.assertGreater(profile["t_ds_solv_kcal_mol"], 0.0)
        self.assertGreater(profile["delta_q_endosomal"], 2.8)

    def test_langevin_optimizer_with_wave_solvation(self) -> None:
        weights = DeNovoObjectiveWeights(w_seq=1.0, w_helix=0.5, w_topo=0.05, w_wave_solv=0.5)
        opt = DeNovoGenerativeOptimizer(weights=weights, max_iterations=8, seed=42)
        res = opt.optimize(sequence_length=10, target_foldstring="HHHHHHHHHH", target_pi=6.8)
        self.assertEqual(len(res.designed_sequence), 10)
        self.assertGreaterEqual(res.loss_breakdown.loss_wave_solv, 0.0)


if __name__ == "__main__":
    unittest.main()
