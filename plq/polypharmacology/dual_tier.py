"""Dual-Tier Lock-Free Concurrency for Massively Parallel Polypharmacology Screening.

Resolves multi-target selectivity and large ternary state matrices without locks:
1. Discrete Topological Tier: supp(r_1) cap supp(r_2) = emptyset => parallel wavefront.
2. Continuous Algebraic Tier: S(r_1) ^ S(r_2) = 0 => concurrent tensor updates.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from plq.dacc.complex import DACCComplex
from plq.gnc.sheaf import CliffordSheaf


@dataclass
class TargetScreeningTask:
    """An atomic screening task evaluating a ligand/binder against a target pocket."""
    task_id: str
    target_name: str
    binder_name: str
    discrete_support: Set[str]
    continuous_sheaf: Optional[CliffordSheaf] = None
    evaluate_fn: Optional[Callable[[], Any]] = None


@dataclass
class TargetScreeningWavefront:
    """A parallel wavefront of tasks guaranteed to have disjoint support closures."""
    wavefront_id: int
    tasks: List[TargetScreeningTask]


class DualTierScheduler:
    """Coordinates lock-free parallel execution across discrete and continuous tiers."""

    @classmethod
    def partition_discrete_wavefronts(
        cls, tasks: List[TargetScreeningTask]
    ) -> List[TargetScreeningWavefront]:
        """Partitions tasks into parallel execution wavefronts based on discrete topological support.
        
        Two tasks t_1 and t_2 can execute in the same wavefront if their support closures are disjoint:
        supp(t_1) cap supp(t_2) = emptyset.
        """
        wavefronts: List[List[TargetScreeningTask]] = []
        wavefront_supports: List[Set[str]] = []

        for task in tasks:
            placed = False
            for idx, wf_supp in enumerate(wavefront_supports):
                if task.discrete_support.isdisjoint(wf_supp):
                    wavefronts[idx].append(task)
                    wf_supp.update(task.discrete_support)
                    placed = True
                    break
            if not placed:
                wavefronts.append([task])
                wavefront_supports.append(set(task.discrete_support))

        return [
            TargetScreeningWavefront(wavefront_id=i, tasks=wf)
            for i, wf in enumerate(wavefronts)
        ]

    @classmethod
    def check_continuous_orthogonality(
        cls, sheaf_a: CliffordSheaf, sheaf_b: CliffordSheaf, tolerance: float = 1e-5
    ) -> bool:
        """Verifies continuous algebraic orthogonality via multivector wedge product.
        
        If S(r_1) ^ S(r_2) = 0, continuous spatial extents do not intersect, allowing
        concurrent lock-free additions to shared binding affinity tensors.
        """
        overlap = sheaf_a.compute_continuous_wedge_overlap(sheaf_b)
        return overlap <= tolerance

    @classmethod
    def execute_screening_parallel(
        cls,
        tasks: List[TargetScreeningTask],
        max_workers: int = 4,
    ) -> List[Any]:
        """Executes screening tasks in lock-free wavefronts using worker threads."""
        wavefronts = cls.partition_discrete_wavefronts(tasks)
        results: List[Any] = []

        for wf in wavefronts:
            # All tasks in a single wavefront have disjoint support: execute in parallel with 0 locks
            with ThreadPoolExecutor(max_workers=min(max_workers, len(wf.tasks) or 1)) as executor:
                futures = []
                for t in wf.tasks:
                    if t.evaluate_fn is not None:
                        futures.append(executor.submit(t.evaluate_fn))
                for f in futures:
                    results.append(f.result())

        return results

    @classmethod
    def partition_wavefronts_theorem_22(
        cls,
        complex_obj: Any,
        rule_specs: List[Tuple[str, Tuple[str, ...], Tuple[str, ...], Optional[Any]]],
    ) -> List[List[str]]:
        """Partitions cellular screening tasks using ToposLang's native Theorem 2.2 Arbiter.
        
        Interference Predicate:
          Interference(r1, r2) = (supp(r1) & supp(r2) != empty) & (S(r1) ^ S(r2) != 0).
        If Interference == False, rules commute identically and execute lock-free.
        """
        try:
            from topos.runtime.wavefront import DualTierWavefrontScheduler, RewriteRule
            topos_rules = [
                RewriteRule(name=name, lhs_cells=lhs, rhs_cells=rhs, blade_state=blade)
                for name, lhs, rhs, blade in rule_specs
            ]
            scheduler = DualTierWavefrontScheduler(complex_obj)
            wavefronts = scheduler.partition_into_wavefronts(topos_rules)
            return [[r.name for r in wf] for wf in wavefronts]
        except Exception:
            # Fallback greedy disjoint partition
            wfs: List[List[str]] = []
            supps: List[Set[str]] = []
            for name, lhs, rhs, _ in rule_specs:
                s = set(lhs) | set(rhs)
                placed = False
                for idx, existing in enumerate(supps):
                    if s.isdisjoint(existing):
                        wfs[idx].append(name)
                        existing.update(s)
                        placed = True
                        break
                if not placed:
                    wfs.append([name])
                    supps.append(set(s))
            return wfs

