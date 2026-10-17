"""ToposLang Symbolic GNC Compiler Pass (--topos-symbolic-gnc).

Lowers PLQ DACC complexes and Clifford sheaves into Topos MLIR dialect
(!topos.cell, !topos.chain, !topos.complex, !topos.multivector), performs compile-time
symbolic blade simplification, zero-blade elimination, and 64/128-byte SIMD packing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from plq.dacc.complex import DACCComplex
from plq.gnc.clifford import Multivector


@dataclass
class SIMDCacheLine:
    """Represents a packed 64-byte (8 x f64) or 128-byte (16 x f64) SIMD register chunk."""
    line_index: int
    blade_indices: List[Tuple[int, ...]]
    values: List[float]
    byte_width: int = 64


@dataclass
class CompiledGNCKernel:
    """Output artifact of the --topos-symbolic-gnc compilation pass."""
    mlir_ir: str
    simd_cache_lines: List[SIMDCacheLine]
    pruned_zero_blades_count: int
    symbolic_flops_reduced: int


class SymbolicGNCPass:
    """Executes the --topos-symbolic-gnc optimization pass."""

    @classmethod
    def lower_to_topos_mlir(cls, complex_obj: DACCComplex, module_name: str = "plq_gnc_module", target: str = "topos") -> str:
        """Emits textual Topos MLIR representation for a DACC cell complex.
        
        Uses ToposLang's MLIREmitter for formal dialect emission, with fallback to native lowering.
        """
        try:
            from topos.compiler.emitter import MLIREmitter
            emitter = MLIREmitter(target=target)
            topos_c = complex_obj.to_topos_complex()
            if target == "topos":
                out = emitter.emit_topos_dialect(topos_c)
            else:
                out = emitter.emit_standard_mlir(topos_c)
            if "topos.gnc_version" not in out:
                out = out.replace("module attributes {", "module attributes {topos.gnc_version = \"2.0.0\", ")
            return out
        except Exception:
            pass


        lines = [
            f"// MLIR Dialect: ToposLang GNC Native Lowering",
            f"// Pass: --topos-symbolic-gnc",
            f"// Module: @{module_name}",
            f"module @{module_name} attributes {{topos.gnc_version = \"2.0.0\"}} {{",
            f"  func.func @evaluate_complex_{complex_obj.name}() -> !topos.complex {{",
            f"    %c0 = topos.create_complex \"{complex_obj.name}\" : !topos.complex",
        ]

        # Lower 0-cells
        for name, cell in sorted(complex_obj.cells[0].items()):
            t_state = cell.attributes.get("ternary_state", 0)
            lines.append(
                f"    topos.attach_cell0 %c0, name(\"{name}\"), ternary_state({t_state}) : !topos.complex"
            )

        # Lower 1-cells
        for name, cell in sorted(complex_obj.cells[1].items()):
            edges = [f"{n}:{c}" for n, c in cell.boundary_coeffs.items()]
            edge_str = ", ".join(edges)
            lines.append(
                f"    topos.attach_cell1 %c0, name(\"{name}\"), boundary([{edge_str}]) : !topos.complex"
            )

        # Lower 2-cells
        for name, cell in sorted(complex_obj.cells[2].items()):
            faces = [f"{e}:{c}" for e, c in cell.boundary_coeffs.items()]
            face_str = ", ".join(faces)
            lines.append(
                f"    topos.attach_cell2 %c0, name(\"{name}\"), boundary([{face_str}]) : !topos.complex"
            )

        lines.append("    topos.verify_nilpotency %c0 : !topos.complex")
        lines.append("    return %c0 : !topos.complex")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)


    @classmethod
    def optimize_multivector(
        cls, mv: Multivector, simd_width_bytes: int = 64
    ) -> Tuple[List[SIMDCacheLine], int, int]:
        """Performs compile-time zero-blade pruning and packs into SIMD cache lines.
        
        simd_width_bytes: 64 bytes (8 doubles) or 128 bytes (16 doubles).
        """
        doubles_per_line = simd_width_bytes // 8
        pruned_count = 0

        active_components: List[Tuple[Tuple[int, ...], float]] = []
        for blade, coeff in mv.components.items():
            if abs(coeff) < 1e-12:
                pruned_count += 1
            else:
                active_components.append((blade, coeff))

        # Pack into SIMD cache lines
        cache_lines: List[SIMDCacheLine] = []
        line_idx = 0
        for i in range(0, len(active_components), doubles_per_line):
            chunk = active_components[i : i + doubles_per_line]
            blades = [item[0] for item in chunk]
            vals = [item[1] for item in chunk]
            # Pad with 0.0 to fill full SIMD register
            while len(vals) < doubles_per_line:
                blades.append(())
                vals.append(0.0)

            cache_lines.append(
                SIMDCacheLine(
                    line_index=line_idx,
                    blade_indices=blades,
                    values=vals,
                    byte_width=simd_width_bytes,
                )
            )
            line_idx += 1

        flops_reduced = pruned_count * 16  # Each eliminated blade saves geometric product flops
        return cache_lines, pruned_count, flops_reduced

    @classmethod
    def compile(
        cls,
        complex_obj: DACCComplex,
        sample_multivectors: Optional[List[Multivector]] = None,
        simd_width_bytes: int = 64,
    ) -> CompiledGNCKernel:
        """Full compilation pipeline executing --topos-symbolic-gnc pass."""
        mlir_text = cls.lower_to_topos_mlir(complex_obj)
        all_lines: List[SIMDCacheLine] = []
        total_pruned = 0
        total_flops = 0

        if sample_multivectors:
            for mv in sample_multivectors:
                lines, pruned, flops = cls.optimize_multivector(mv, simd_width_bytes)
                all_lines.extend(lines)
                total_pruned += pruned
                total_flops += flops

        return CompiledGNCKernel(
            mlir_ir=mlir_text,
            simd_cache_lines=all_lines,
            pruned_zero_blades_count=total_pruned,
            symbolic_flops_reduced=total_flops,
        )
