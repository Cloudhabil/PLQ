# Pharmacological Lattice Quantization (PLQ) v2.0
## ToposLang ($\tau$-Lang) and Geometric Native Coding (GNC) Native Framework

Pharmacological Lattice Quantization (PLQ) transforms conventional drug discovery and protein design pipelines from standard CPU/Python flat vectors and heavy matrix transforms into a lock-free, coordinate-free, and topologically verified execution system.

---

## Architecture Overview

```
PLQ/
├── plq/
│   ├── dacc/                 # Directed Acyclic Cell Complexes & Homology Fingerprinting
│   │   ├── complex.py        # MolecularDACC, PocketDACC, boundary nilpotency (d² = 0)
│   │   └── fingerprint.py    # Topological invariant signatures (β₀, β₁, β₂, χ)
│   ├── gnc/                  # Geometric Native Coding & Clifford Multivector Sheaves
│   │   ├── clifford.py       # PGA (ℝ³,⁰,¹) & CGA (ℝ⁴,¹,⁰) multivector blade arithmetic
│   │   ├── versor.py         # Coordinate-free versor motors (A' = M A M⁻¹) & rigid docking
│   │   └── sheaf.py          # Multivector cellular sheaves & continuous wedge (S₁ ∧ S₂ = 0)
│   ├── ternary/              # PLQ Ternary Lattice & Histidine Switch Dynamics
│   │   ├── lattice.py        # Ternary state space T ∈ {-1, 0, +1} & interaction matrices
│   │   └── his_switch.py     # Bifold manifold: pH 6.5 (ON, His⁺) vs. pH 7.4 (OFF, His⁰)
│   ├── polypharmacology/     # Massively Parallel Polypharmacology Screening
│   │   └── dual_tier.py      # Dual-tier lock-free concurrency scheduler
│   ├── hodge/                # Deterministic Generative Convergence via Hodge Theory
│   │   ├── laplacian.py      # Combinatorial Hodge Laplacian (L₁ = B₁ᵀ B₁ + B₂ B₂ᵀ ∈ ℚ)
│   │   └── engine.py         # Hodge-guided MCMC crossover (until betti(K, dim=1) == 0)
│   └── compiler/             # Compiler-Level Acceleration
│       └── symbolic_gnc.py   # --topos-symbolic-gnc pass lowering to MLIR & 64-byte SIMD packing
└── tests/
    └── test_plq_core.py      # 11 unit & integration tests covering all PLQ modules
```

---

## Key Mathematical Capabilities

1. **Topological Data Structures:** Directed Acyclic Cell Complexes $\mathcal{K} = (K, \partial)$ with statically enforced nilpotency $\partial_{k-1} \circ \partial_k \equiv 0$.
2. **Coordinate-Free Geometric Algebra:** 3D Projective Geometric Algebra (PGA) and Conformal Geometric Algebra (CGA) replacing 3x3 rotation matrices with versor sandwich operators $A' = M A M^{-1}$.
3. **Dual-Tier Lock-Free Concurrency:**
   - Discrete Topological Tier: $\text{supp}(t_1) \cap \text{supp}(t_2) = \emptyset$ dispatches parallel wavefronts with zero lock contention.
   - Continuous Algebraic Tier: $S_1 \wedge S_2 = 0$ ensures non-intersecting continuous support for lock-free affinity updates.
4. **Hodge-Guided Loop Termination:** Combinatorial Hodge Laplacian kernel evaluation over $\mathbb{Q}$ terminates de novo assembly when $\beta_1 = \dim(\ker L_1) = 0$.
5. **Symbolic GNC Pass:** Lowers to Topos MLIR dialect (`!topos.cell`, `!topos.complex`), prunes zero-blades, and packs multivector components into 64-byte/128-byte SIMD registers.

---

## Test Execution

Run the complete test suite:
```bash
python3 -m unittest discover tests
```
All tests pass 100% with zero external dependencies.
