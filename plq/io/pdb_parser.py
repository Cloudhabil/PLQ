"""PDB and SDF Molecular Structure Ingestion Engine into 8D DACC Complexes.

Translates experimental structural data (crystallography, Cryo-EM, AlphaFold PDBs,
and small-molecule SDF files) into graded Directed Acyclic Cell Complexes (DACC)
embedded in the 8-dimensional Pharmacological Configuration Manifold M^8:
    x = (x1, x2, x3, phi, psi, pH, rho_solv, t)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from plq.dacc.complex import DACCComplex


@dataclass
class AtomRecord8D:
    """An atom representation with full 8D pharmacological manifold state."""
    atom_id: str
    name: str
    element: str
    resname: str
    resid: int
    chain_id: str
    x: float
    y: float
    z: float
    phi: float = 0.0          # Backbone dihedral torsion phi (radians)
    psi: float = 0.0          # Backbone dihedral torsion psi (radians)
    ph: float = 7.4           # Environmental pH
    rho_solv: float = 1.0     # Solvent accessibility / hydration density
    t_param: float = 0.0      # Allosteric conformational trajectory parameter
    charge: float = 0.0       # Formal / ternary partial charge

    @property
    def coordinates_8d(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """Returns the canonical 8D coordinates tuple."""
        return (
            self.x,
            self.y,
            self.z,
            self.phi,
            self.psi,
            self.ph,
            self.rho_solv,
            self.t_param,
        )


@dataclass
class MolecularStructure8D:
    """Parsed molecular structure with 8D atoms, bonds, and rings."""
    name: str
    atoms: List[AtomRecord8D]
    bonds: List[Tuple[str, str, int]] = field(default_factory=list)  # (atom_a, atom_b, bond_order)
    rings: List[List[str]] = field(default_factory=list)             # cycles of atom_ids
    cavities: List[Tuple[str, str, str, str]] = field(default_factory=list)  # 4-tuples of pocket atoms


class PDBParser:
    """Parses PDB and SDF streams into DACCComplex with 8D coordinate embeddings."""

    # Standard covalent radii (Angstroms) for bond detection
    COVALENT_RADII: Dict[str, float] = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "S": 1.05,
        "P": 1.07,
        "F": 0.57,
        "CL": 1.02,
        "BR": 1.20,
        "I": 1.39,
    }

    # Ternary amino acid charges at physiological pH 7.4 vs acidic pH 6.0
    AMINO_ACID_CHARGES: Dict[str, Tuple[float, float]] = {
        # Resname: (charge_at_7.4, charge_at_6.0)
        "ASP": (-1.0, -1.0),
        "GLU": (-1.0, -1.0),
        "LYS": (+1.0, +1.0),
        "ARG": (+1.0, +1.0),
        "HIS": (0.0, +1.0),  # Histidine pH switch
        "TYR": (0.0, 0.0),
        "CYS": (0.0, 0.0),
    }

    @classmethod
    def _dihedral(
        cls,
        p0: Tuple[float, float, float],
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        p3: Tuple[float, float, float],
    ) -> float:
        """Computes dihedral angle (in radians) between 4 consecutive points."""
        b0 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        b1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        b2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])

        # Normalize b1
        len_b1 = math.sqrt(b1[0]**2 + b1[1]**2 + b1[2]**2)
        if len_b1 < 1e-9:
            return 0.0
        b1_norm = (b1[0] / len_b1, b1[1] / len_b1, b1[2] / len_b1)

        # Vector cross products: v = b0 x b1, w = b1 x b2
        v = (
            b0[1] * b1[2] - b0[2] * b1[1],
            b0[2] * b1[0] - b0[0] * b1[2],
            b0[0] * b1[1] - b0[1] * b1[0],
        )
        w = (
            b1[1] * b2[2] - b1[2] * b2[1],
            b1[2] * b2[0] - b1[0] * b2[2],
            b1[0] * b2[1] - b1[1] * b2[0],
        )

        x = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]
        # Cross product of v and w
        vw_cross = (
            v[1] * w[2] - v[2] * w[1],
            v[2] * w[0] - v[0] * w[2],
            v[0] * w[1] - v[1] * w[0],
        )
        y = vw_cross[0] * b1_norm[0] + vw_cross[1] * b1_norm[1] + vw_cross[2] * b1_norm[2]

        return math.atan2(y, x)

    @classmethod
    def parse_pdb_string(
        cls,
        pdb_content: str,
        name: str = "PDB_Complex",
        ph: float = 7.4,
        t_param: float = 0.0,
        contact_cutoff: float = 4.5,
    ) -> DACCComplex:
        """Parses a PDB string into a DACCComplex with 8D coordinate embeddings."""
        atoms: List[AtomRecord8D] = []
        atom_by_res: Dict[int, Dict[str, AtomRecord8D]] = {}

        for line in pdb_content.splitlines():
            record = line[0:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue

            try:
                atom_id = line[6:11].strip()
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain_id = line[21:22].strip() or "A"
                resid = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
                if not element or element.isdigit():
                    element = atom_name[0]
            except (ValueError, IndexError):
                continue

            # Determine charge at current pH
            charges = cls.AMINO_ACID_CHARGES.get(resname, (0.0, 0.0))
            charge = charges[0] if ph >= 6.8 else charges[1]

            atom_rec = AtomRecord8D(
                atom_id=f"{resname}_{resid}_{atom_name}_{atom_id}",
                name=atom_name,
                element=element.upper(),
                resname=resname,
                resid=resid,
                chain_id=chain_id,
                x=x,
                y=y,
                z=z,
                ph=ph,
                t_param=t_param,
                charge=charge,
            )
            atoms.append(atom_rec)

            if resid not in atom_by_res:
                atom_by_res[resid] = {}
            atom_by_res[resid][atom_name] = atom_rec

        # Compute backbone dihedrals phi and psi
        sorted_resids = sorted(atom_by_res.keys())
        for idx, r in enumerate(sorted_resids):
            curr_atoms = atom_by_res[r]
            if "N" in curr_atoms and "CA" in curr_atoms and "C" in curr_atoms:
                p_n = (curr_atoms["N"].x, curr_atoms["N"].y, curr_atoms["N"].z)
                p_ca = (curr_atoms["CA"].x, curr_atoms["CA"].y, curr_atoms["CA"].z)
                p_c = (curr_atoms["C"].x, curr_atoms["C"].y, curr_atoms["C"].z)

                # phi = C_{i-1} - N_i - CA_i - C_i
                if idx > 0 and "C" in atom_by_res[sorted_resids[idx - 1]]:
                    p_c_prev = (
                        atom_by_res[sorted_resids[idx - 1]]["C"].x,
                        atom_by_res[sorted_resids[idx - 1]]["C"].y,
                        atom_by_res[sorted_resids[idx - 1]]["C"].z,
                    )
                    phi_val = cls._dihedral(p_c_prev, p_n, p_ca, p_c)
                    for a in curr_atoms.values():
                        a.phi = phi_val

                # psi = N_i - CA_i - C_i - N_{i+1}
                if idx < len(sorted_resids) - 1 and "N" in atom_by_res[sorted_resids[idx + 1]]:
                    p_n_next = (
                        atom_by_res[sorted_resids[idx + 1]]["N"].x,
                        atom_by_res[sorted_resids[idx + 1]]["N"].y,
                        atom_by_res[sorted_resids[idx + 1]]["N"].z,
                    )
                    psi_val = cls._dihedral(p_n, p_ca, p_c, p_n_next)
                    for a in curr_atoms.values():
                        a.psi = psi_val

        # Build DACC Complex
        complex_obj = DACCComplex(name=name)

        # 0-Cells
        for a in atoms:
            complex_obj.add_0cell(
                a.atom_id,
                element=a.element,
                resname=a.resname,
                resid=a.resid,
                coordinates_8d=a.coordinates_8d,
                charge=a.charge,
                ph=a.ph,
                phi=a.phi,
                psi=a.psi,
            )

        # 1-Cells (Covalent bonds and distance contact channels)
        num_atoms = len(atoms)
        edge_id = 0
        added_edges: Set[Tuple[str, str]] = set()

        for i in range(num_atoms):
            ai = atoms[i]
            ri = cls.COVALENT_RADII.get(ai.element, 0.77)
            for j in range(i + 1, num_atoms):
                aj = atoms[j]
                rj = cls.COVALENT_RADII.get(aj.element, 0.77)
                dist = math.sqrt((ai.x - aj.x)**2 + (ai.y - aj.y)**2 + (ai.z - aj.z)**2)

                is_covalent = dist <= (ri + rj + 0.45)
                is_contact = dist <= contact_cutoff

                if is_covalent or is_contact:
                    pair_key = tuple(sorted((ai.atom_id, aj.atom_id)))
                    if pair_key not in added_edges:
                        added_edges.add(pair_key)
                        ename = f"e_{edge_id}_{ai.atom_id}__{aj.atom_id}"
                        complex_obj.add_1cell(
                            ename,
                            source=ai.atom_id,
                            target=aj.atom_id,
                            distance=round(dist, 4),
                            is_covalent=is_covalent,
                        )
                        edge_id += 1

        # 2-Cells (Aromatic ring patches and peptide planar units)
        ring_face_id = 0
        for r in sorted_resids:
            ca = atom_by_res[r]
            # Detect aromatic rings
            aromatic_sets = {
                "HIS": ["CG", "ND1", "CE1", "NE2", "CD2"],
                "PHE": ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
                "TYR": ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
                "TRP": ["CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"],
            }
            if ca.get("CA") and ca["CA"].resname in aromatic_sets:
                ring_atoms = aromatic_sets[ca["CA"].resname]
                if all(ra in ca for ra in ring_atoms):
                    # Form oriented 1-cycle around ring
                    cycle_edges = []
                    ring_len = len(ring_atoms)
                    valid_cycle = True
                    for k in range(ring_len):
                        src = ca[ring_atoms[k]].atom_id
                        tgt = ca[ring_atoms[(k + 1) % ring_len]].atom_id
                        # Find matching 1-cell
                        found_edge = None
                        coeff = 1
                        for e_name, e_cell in complex_obj.cells[1].items():
                            if e_cell.boundary_coeffs.get(tgt) == 1 and e_cell.boundary_coeffs.get(src) == -1:
                                found_edge = e_name
                                coeff = 1
                                break
                            elif e_cell.boundary_coeffs.get(src) == 1 and e_cell.boundary_coeffs.get(tgt) == -1:
                                found_edge = e_name
                                coeff = -1
                                break
                        if found_edge:
                            cycle_edges.append((found_edge, coeff))
                        else:
                            # Add bridging bond if missing
                            new_e = f"e_ring_{ring_face_id}_{src}__{tgt}"
                            complex_obj.add_1cell(new_e, source=src, target=tgt, distance=1.4, is_covalent=True)
                            cycle_edges.append((new_e, 1))

                    if valid_cycle and len(cycle_edges) >= 3:
                        try:
                            complex_obj.add_2cell(
                                f"f_aromatic_{ring_face_id}_{ca['CA'].resname}_{r}",
                                cycle_edges,
                                verify_nilpotency=True,
                                ring_type="aromatic",
                            )
                            ring_face_id += 1
                        except Exception:
                            pass

        return complex_obj

    @classmethod
    def parse_pdb_file(cls, filepath: str | Path, **kwargs: Any) -> DACCComplex:
        """Reads and parses a PDB file on disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"PDB file not found: {path}")
        content = path.read_text(encoding="utf-8")
        name = kwargs.pop("name", path.stem)
        return cls.parse_pdb_string(content, name=name, **kwargs)

    @classmethod
    def parse_sdf_string(cls, sdf_content: str, name: str = "SDF_Ligand") -> DACCComplex:
        """Parses an MDL MOL/SDF V2000 string into a DACCComplex."""
        lines = sdf_content.strip().splitlines()
        if len(lines) < 4:
            raise ValueError("Invalid SDF format: header too short")

        mol_name = lines[0].strip() or name
        counts_line = lines[3]
        try:
            num_atoms = int(counts_line[0:3].strip())
            num_bonds = int(counts_line[3:6].strip())
        except (ValueError, IndexError):
            raise ValueError("Invalid SDF V2000 counts line")

        complex_obj = DACCComplex(name=mol_name)
        atom_names: List[str] = []

        atom_start = 4
        for i in range(num_atoms):
            aline = lines[atom_start + i]
            x = float(aline[0:10].strip())
            y = float(aline[10:20].strip())
            z = float(aline[20:30].strip())
            elem = aline[31:34].strip()
            aname = f"{elem}_{i+1}"
            atom_names.append(aname)
            complex_obj.add_0cell(
                aname,
                element=elem,
                coordinates_8d=(x, y, z, 0.0, 0.0, 7.4, 1.0, 0.0),
            )

        bond_start = atom_start + num_atoms
        for j in range(num_bonds):
            bline = lines[bond_start + j]
            a1 = int(bline[0:3].strip()) - 1
            a2 = int(bline[3:6].strip()) - 1
            order = int(bline[6:9].strip())
            if 0 <= a1 < num_atoms and 0 <= a2 < num_atoms:
                ename = f"b_{j+1}_{atom_names[a1]}__{atom_names[a2]}"
                complex_obj.add_1cell(
                    ename,
                    source=atom_names[a1],
                    target=atom_names[a2],
                    bond_order=order,
                    is_covalent=True,
                )

        return complex_obj


def parse_pdb(source: str, **kwargs: Any) -> DACCComplex:
    """Convenience function to parse a PDB file path or PDB string."""
    from pathlib import Path
    p = Path(source)
    if p.exists() and p.is_file():
        return PDBParser.parse_pdb_file(str(p), **kwargs)
    return PDBParser.parse_pdb_string(source, **kwargs)

