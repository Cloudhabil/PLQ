"""ToposLang Contract Bridge to PLQ DACC Runtime.

Parses ToposLang cellular contract declarations (.tau) as specified in
Chapter 3 of the monograph, lowering them to DACCComplex objects and executing
compile-time invariant assertions:
    assert nilpotency == true;
    assert betti(1) == 0;
    assert euler_characteristic == 1;
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from plq.dacc.complex import DACCComplex, NilpotencyViolationError
from plq.dacc.fingerprint import TopologicalDACCInvariant


@dataclass
class ContractVerificationReport:
    """Report emitted from compiling and verifying a ToposLang contract."""
    contract_name: str
    num_0cells: int
    num_1cells: int
    num_2cells: int
    num_3cells: int
    beta_0: int
    beta_1: int
    beta_2: int
    euler_characteristic: int
    nilpotency_verified: bool
    all_assertions_passed: bool
    diagnostic_trace: List[str]


class ToposContractBridge:
    """Parses and compiles ToposLang contract source strings into DACCComplexes."""

    @classmethod
    def compile_contract(cls, source_code: str) -> Tuple[DACCComplex, ContractVerificationReport]:
        """Parses ToposLang declarative contract and verifies boundary nilpotency and Betti assertions."""
        trace: List[str] = []
        name_match = re.search(r'(?:epitope|contract|target)\s+([A-Za-z0-9_]+)', source_code)
        contract_name = name_match.group(1) if name_match else "ToposContract"
        complex_obj = DACCComplex(name=contract_name)

        # Try native ToposLang Lexer and Parser first
        try:
            from topos.frontend.lexer import Lexer
            from topos.frontend.parser import Parser
            from topos.frontend.ast import (
                Cell0Decl,
                Cell1Decl,
                Cell2Decl,
                AssertNilpotencyStmt,
                BettiCondition,
            )

            lexer = Lexer(source_code)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast_program = parser.parse()

            for stmt in ast_program.statements:
                if isinstance(stmt, Cell0Decl):
                    complex_obj.add_0cell(stmt.name)
                    trace.append(f"AST 0-cell: {stmt.name}")
                elif isinstance(stmt, Cell1Decl):
                    if stmt.source not in complex_obj.cells[0]:
                        complex_obj.add_0cell(stmt.source)
                    if stmt.target not in complex_obj.cells[0]:
                        complex_obj.add_0cell(stmt.target)
                    complex_obj.add_1cell(stmt.name, source=stmt.source, target=stmt.target)
                    trace.append(f"AST 1-cell: {stmt.name} ({stmt.source} -> {stmt.target})")
                elif isinstance(stmt, Cell2Decl):
                    edge_tuples = [(term.name, int(term.coeff)) for term in stmt.boundary.terms]
                    complex_obj.add_2cell(stmt.name, edge_tuples, verify_nilpotency=True)
                    trace.append(f"AST 2-cell: {stmt.name}")

        except Exception as e:
            trace.append(f"Native AST parser fallback: {e}")
            # Fallback regex parser for legacy/simplified declarations
            # 1. Parse 0-cells: cell 0 <name>; or cell0 <name>;
            c0_matches = re.findall(r'cell(?:0|\s+0)\s+([A-Za-z0-9_]+)', source_code)
            for name in c0_matches:
                if name not in complex_obj.cells[0]:
                    complex_obj.add_0cell(name)
                    trace.append(f"Registered 0-cell: {name}")

            # 2. Parse 1-cells: cell 1 <name> : <src> -> <tgt>;
            c1_matches = re.findall(r'cell(?:1|\s+1)\s+([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)', source_code)
            for e_name, src, tgt in c1_matches:
                if src not in complex_obj.cells[0]:
                    complex_obj.add_0cell(src)
                if tgt not in complex_obj.cells[0]:
                    complex_obj.add_0cell(tgt)
                if e_name not in complex_obj.cells[1]:
                    complex_obj.add_1cell(e_name, source=src, target=tgt)
                    trace.append(f"Registered 1-cell: {e_name} ({src} -> {tgt})")

            # 3. Parse 2-cells: cell 2 <name> { <edges> };
            c2_matches = re.finditer(r'cell(?:2|\s+2)\s+([A-Za-z0-9_]+)\s*\{([^}]+)\}', source_code)
            for m in c2_matches:
                f_name = m.group(1)
                body = m.group(2)
                edge_pairs = []
                for item in body.split(','):
                    item = item.strip()
                    if ':' in item:
                        e_id, coeff_str = item.split(':')
                        edge_pairs.append((e_id.strip(), int(coeff_str.strip())))
                try:
                    if f_name not in complex_obj.cells[2]:
                        complex_obj.add_2cell(f_name, edge_pairs, verify_nilpotency=True)
                        trace.append(f"Registered 2-cell: {f_name} (Nilpotency verified)")
                except NilpotencyViolationError as err:
                    trace.append(f"ERROR in 2-cell {f_name}: {err}")


        # 4. Invariant assertions evaluation
        sig = TopologicalDACCInvariant.compute_signature(complex_obj)
        nilpotency_ok = True
        try:
            complex_obj.verify_all_nilpotence()
        except NilpotencyViolationError:
            nilpotency_ok = False

        assertions_passed = True
        if "assert nilpotency == true" in source_code:
            if not nilpotency_ok:
                assertions_passed = False
                trace.append("ASSERTION FAILED: assert nilpotency == true")

        betti1_match = re.search(r'assert\s+betti\(\s*1\s*\)\s*==\s*(\d+)', source_code)
        if betti1_match:
            expected_b1 = int(betti1_match.group(1))
            if sig.beta_1 != expected_b1:
                assertions_passed = False
                trace.append(f"ASSERTION FAILED: assert betti(1) == {expected_b1} (got {sig.beta_1})")

        chi_match = re.search(r'assert\s+euler_characteristic\s*==\s*(\d+)', source_code)
        if chi_match:
            expected_chi = int(chi_match.group(1))
            if sig.chi != expected_chi:
                assertions_passed = False
                trace.append(f"ASSERTION FAILED: assert euler_characteristic == {expected_chi} (got {sig.chi})")

        report = ContractVerificationReport(
            contract_name=contract_name,
            num_0cells=sig.num_0cells,
            num_1cells=sig.num_1cells,
            num_2cells=sig.num_2cells,
            num_3cells=sig.num_3cells,
            beta_0=sig.beta_0,
            beta_1=sig.beta_1,
            beta_2=sig.beta_2,
            euler_characteristic=sig.chi,
            nilpotency_verified=nilpotency_ok,
            all_assertions_passed=assertions_passed,
            diagnostic_trace=trace,
        )

        return complex_obj, report
