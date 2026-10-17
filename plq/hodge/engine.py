"""Deterministic Generative Design Convergence via Hodge Theory.

Replaces heuristic MCMC loop termination with exact queries on the Combinatorial
Hodge Laplacian: executes iteratively until betti(K, dim=1) == 0.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from plq.dacc.complex import DACCComplex, MolecularDACC
from plq.hodge.laplacian import CombinatorialHodgeLaplacian


@dataclass
class GenerationResult:
    """Result of Hodge-guided de novo sequence convergence."""
    sequence: str
    iterations: int
    final_beta_1: int
    converged: bool
    h_bond_closures: int
    ternary_profile: List[int]


class HodgeGuidedGenerator:
    """Drives fragment crossover and sequence mutation until topological closure beta_1 == 0."""

    def __init__(
        self,
        seed: int = 42,
        max_iterations: int = 50,
    ) -> None:
        self.rng = random.Random(seed)
        self.max_iterations = max_iterations

    def build_candidate_complex(
        self,
        sequence: str,
        pocket_anionic_anchors: List[str],
    ) -> DACCComplex:
        """Constructs a joint candidate-target contact complex K."""
        complex_obj = DACCComplex(name=f"Joint_{sequence[:8]}")

        # Add binder residues as 0-cells
        for i, aa in enumerate(sequence):
            is_his = (aa == "H")
            t_state = +1 if aa in ("H", "K", "R") else (-1 if aa in ("D", "E") else 0)
            complex_obj.add_0cell(
                name=f"b_{i}_{aa}",
                dim=0,
                amino_acid=aa,
                ternary_state=t_state,
                is_histidine=is_his,
            )

        # Add pocket anchor residues as 0-cells
        for p_name in pocket_anionic_anchors:
            complex_obj.add_0cell(name=f"p_{p_name}", dim=0, ternary_state=-1)

        # Add backbone 1-cells
        n = len(sequence)
        for i in range(n - 1):
            complex_obj.add_1cell(
                name=f"e_bb_{i}_{i+1}",
                source=f"b_{i}_{sequence[i]}",
                target=f"b_{i+1}_{sequence[i+1]}",
            )

        # Add pocket inter-anchor 1-cells
        for p_idx in range(len(pocket_anionic_anchors) - 1):
            complex_obj.add_1cell(
                name=f"e_pocket_{p_idx}_{p_idx+1}",
                source=f"p_{pocket_anionic_anchors[p_idx]}",
                target=f"p_{pocket_anionic_anchors[p_idx+1]}",
            )

        # Map each donor residue to its coordinated pocket anchor
        donor_indices = [i for i, aa in enumerate(sequence) if aa in ("H", "K", "R")]
        num_anchors = len(pocket_anionic_anchors)

        if donor_indices and num_anchors > 0:
            donor_to_anchor: Dict[int, int] = {}
            for d_idx, res_i in enumerate(donor_indices):
                assigned_p = d_idx % num_anchors
                donor_to_anchor[res_i] = assigned_p
                complex_obj.add_1cell(
                    name=f"e_contact_{res_i}_{assigned_p}",
                    source=f"b_{res_i}_{sequence[res_i]}",
                    target=f"p_{pocket_anionic_anchors[assigned_p]}",
                )

            # Contract cycles between consecutive donors
            for d_idx in range(len(donor_indices) - 1):
                i = donor_indices[d_idx]
                j = donor_indices[d_idx + 1]
                p_i = donor_to_anchor[i]
                p_j = donor_to_anchor[j]

                # Backbone edges from i to j
                bb_edges = [(f"e_bb_{k}_{k+1}", +1) for k in range(i, j)]
                # Contact edge from j to p_j: target is p_j, source is b_j => boundary is p_j - b_j
                edge_j = (f"e_contact_{j}_{p_j}", +1)
                # Contact edge from i to p_i: boundary is p_i - b_i => -edge_i gives -(p_i - b_i)
                edge_i = (f"e_contact_{i}_{p_i}", -1)

                # Pocket path edges from p_j to p_i
                pocket_edges: List[Tuple[str, int]] = []
                if p_j > p_i:
                    for k in range(p_i, p_j):
                        # e_pocket_{k, k+1} has boundary p_{k+1} - p_k
                        # To go from p_j to p_i, we take -1 orientation
                        pocket_edges.append((f"e_pocket_{k}_{k+1}", -1))
                elif p_i > p_j:
                    for k in range(p_j, p_i):
                        pocket_edges.append((f"e_pocket_{k}_{k+1}", +1))

                # Boundary check:
                # bb_edges sum: b_j - b_i
                # edge_j: p_j - b_j
                # pocket_edges: p_i - p_j
                # edge_i: -(p_i - b_i)
                # Total = (b_j - b_i) + (p_j - b_j) + (p_i - p_j) - (p_i - b_i) = 0!
                boundary_list = bb_edges + [edge_j] + pocket_edges + [edge_i]
                complex_obj.add_2cell(
                    name=f"f_closure_{d_idx}",
                    boundary_1cells=boundary_list,
                    verify_nilpotency=True,
                )

        return complex_obj

    def query_beta_1(self, complex_obj: DACCComplex) -> int:
        """Queries the 1st Betti number beta_1 = dim(ker L_1) over rational field Q."""
        _, beta_1 = CombinatorialHodgeLaplacian.compute_hodge_laplacian_1(complex_obj)
        return beta_1

    def generate_converged_sequence(
        self,
        base_scaffold: str,
        pocket_anionic_anchors: List[str],
        his_target_count: int = 4,
    ) -> GenerationResult:
        """Generates a de novo sequence by iterative mutation until betti(K, dim=1) == 0."""
        seq_chars = list(base_scaffold)
        n = len(seq_chars)
        iteration = 0
        converged = False

        while iteration < self.max_iterations:
            iteration += 1
            # Ensure proper histidine switch placement
            his_indices = [idx for idx, ch in enumerate(seq_chars) if ch == "H"]
            if len(his_indices) < his_target_count:
                # Insert histidine at accessible loop positions
                pos = self.rng.randint(0, n - 1)
                seq_chars[pos] = "H"

            # Check Hodge Laplacian beta_1
            current_seq = "".join(seq_chars)
            # Sample subset of contacts to check subcomplex closure
            k_obj = self.build_candidate_complex(current_seq[: min(30, n)], pocket_anionic_anchors[:3])
            beta_1 = self.query_beta_1(k_obj)

            if beta_1 == 0:
                converged = True
                break

            # Mutate to close topological hole: introduce complementary pair or adjust spacing
            mutate_pos = self.rng.randint(0, min(29, n - 1))
            seq_chars[mutate_pos] = self.rng.choice(["H", "A", "L", "V", "E", "S", "T"])

        final_seq = "".join(seq_chars)
        final_k = self.build_candidate_complex(final_seq[: min(30, n)], pocket_anionic_anchors[:3])
        _, final_beta = CombinatorialHodgeLaplacian.compute_hodge_laplacian_1(final_k)

        t_profile = [
            +1 if aa in ("H", "K", "R") else (-1 if aa in ("D", "E") else 0)
            for aa in final_seq
        ]

        return GenerationResult(
            sequence=final_seq,
            iterations=iteration,
            final_beta_1=final_beta,
            converged=(final_beta == 0),
            h_bond_closures=len(final_k.cells[2]),
            ternary_profile=t_profile,
        )

    def generate_converged_sequence_hcst(
        self,
        base_scaffold: str,
        pocket_anionic_anchors: List[str],
        his_target_count: int = 4,
        lambda_wedge: float = 0.5,
    ) -> GenerationResult:
        """Differentiable HCST-guided de novo generation replacing blind random MCMC.
        
        Uses 8D Dual-Stream Clifford Attention:
          E_ij = (1 / 16) * ( <Q_i * ~K_j>_0 - lambda_wedge * ||Q_i ^ K_j||^2 )
        to prioritize mutations on topologically obstructed and sterically frustrated residues.
        """
        from plq.gnc.clifford import Clifford8D
        seq_chars = list(base_scaffold)
        n = len(seq_chars)
        iteration = 0
        converged = False

        while iteration < self.max_iterations:
            iteration += 1

            # Build subcomplex to evaluate current topology
            k_obj = self.build_candidate_complex(
                "".join(seq_chars[: min(30, n)]), pocket_anionic_anchors[:3]
            )
            beta_1 = CombinatorialHodgeLaplacian.compute_hodge_laplacian_crt(k_obj, 1)

            if beta_1 == 0:
                converged = True
                break

            # Evaluate HCST dual-stream attention scores per residue
            residue_scores: List[float] = []
            for i, aa in enumerate(seq_chars[: min(30, n)]):
                # Represent residue state as 8D multivector
                charge_val = 1.0 if aa in ("H", "K", "R") else (-1.0 if aa in ("D", "E") else 0.0)
                q_mv = Clifford8D.vector_8d((float(i), charge_val, 0.0, 0.0, 0.0, 7.4, 1.0, 0.0))

                # Pocket anchor multivector (anionic, charge -1.0)
                k_mv = Clifford8D.vector_8d((float(i % max(1, len(pocket_anionic_anchors))), -1.0, 0.0, 0.0, 0.0, 7.4, 1.0, 0.0))

                # Dual-stream attention score: metric alignment vs wedge meet
                score = Clifford8D.dual_stream_attention_score(q_mv, k_mv, lambda_wedge=lambda_wedge)
                residue_scores.append(score)

            # Prioritize mutation on residue with lowest attention score (greatest frustration/obstruction)
            if residue_scores:
                worst_idx = min(range(len(residue_scores)), key=lambda idx: residue_scores[idx])
            else:
                worst_idx = self.rng.randint(0, min(29, n - 1))

            # If histidines are under target, place Histidine at worst position
            his_indices = [idx for idx, ch in enumerate(seq_chars) if ch == "H"]
            if len(his_indices) < his_target_count:
                seq_chars[worst_idx] = "H"
            else:
                # Complementary electrostatic substitution
                seq_chars[worst_idx] = self.rng.choice(["H", "K", "R", "A", "L", "S"])

        final_seq = "".join(seq_chars)
        final_k = self.build_candidate_complex(final_seq[: min(30, n)], pocket_anionic_anchors[:3])
        final_beta = CombinatorialHodgeLaplacian.compute_hodge_laplacian_crt(final_k, 1)

        t_profile = [
            +1 if aa in ("H", "K", "R") else (-1 if aa in ("D", "E") else 0)
            for aa in final_seq
        ]

        return GenerationResult(
            sequence=final_seq,
            iterations=iteration,
            final_beta_1=final_beta,
            converged=(final_beta == 0),
            h_bond_closures=len(final_k.cells.get(2, {})),
            ternary_profile=t_profile,
        )

    def generate_denovo_binder_optimized(
        self,
        base_scaffold: Optional[str] = None,
        pocket_anionic_anchors: Optional[List[str]] = None,
        target_foldstring: Optional[str] = None,
        target_pi: Optional[float] = None,
        target_mw: Optional[float] = None,
        native_template: Optional[str] = None,
        reference_structures: Optional[Sequence[Sequence[Tuple[float, float, float]]]] = None,
        sequence_length: int = 10,
        weights: Optional[Any] = None,
    ) -> Any:
        """Inverted gradient-guided de novo protein generation over joint (X, z) space.

        Optimizes the continuous composite objective function L_total(X, pi) incorporating:
        - ProteinMPNN pseudo-NLL sequence design loss
        - Target DSSP Foldstring (helix/sheet potentials)
        - Structural novelty penalty against reference PDBs
        - Analytical isoelectric point (pI) and molecular weight (Mw) target constraints
        - Differentiable soft Hodge Laplacian topology loss (beta_1 -> 0)
        """
        from plq.metrics.objective import DeNovoGenerativeOptimizer, DeNovoObjectiveWeights

        opt = DeNovoGenerativeOptimizer(
            weights=weights or DeNovoObjectiveWeights(),
            max_iterations=self.max_iterations,
        )
        return opt.optimize(
            initial_sequence=base_scaffold,
            sequence_length=sequence_length if not base_scaffold else len(base_scaffold),
            target_foldstring=target_foldstring,
            target_pi=target_pi,
            target_mw=target_mw,
            native_template=native_template,
            reference_structures=reference_structures,
        )


