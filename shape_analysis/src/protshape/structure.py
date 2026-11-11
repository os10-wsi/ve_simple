"""Reading AlphaFold PDB models into numpy arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scales import ATOMIC_MASS, VDW_RADIUS

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "SEC": "U", "MSE": "M",
}


@dataclass
class Structure:
    coords: np.ndarray        # (n_atoms, 3)
    element: np.ndarray       # (n_atoms,)
    atom_name: np.ndarray     # (n_atoms,)
    resname: np.ndarray       # (n_atoms,) residue name per atom
    res_index: np.ndarray     # (n_atoms,) 0-based index into the residue arrays
    resnum: np.ndarray        # (n_res,) residue numbers from the model (1-based UniProt numbering in AFDB)
    res_names: np.ndarray     # (n_res,)
    plddt: np.ndarray         # (n_res,) from the B-factor column of CA
    ca_index: np.ndarray      # (n_res,) atom index of CA (-1 if missing)

    @property
    def n_res(self) -> int:
        return len(self.resnum)

    @property
    def sequence(self) -> str:
        return "".join(THREE_TO_ONE.get(r, "X") for r in self.res_names)

    @property
    def radii(self) -> np.ndarray:
        return np.array([VDW_RADIUS.get(e, 1.8) for e in self.element])

    @property
    def masses(self) -> np.ndarray:
        return np.array([ATOMIC_MASS.get(e, 12.0) for e in self.element])

    def subset(self, residue_mask: np.ndarray) -> "Structure":
        """Structure restricted to residues where residue_mask is True."""
        keep_res = np.flatnonzero(residue_mask)
        remap = -np.ones(self.n_res, dtype=int)
        remap[keep_res] = np.arange(len(keep_res))
        atom_keep = residue_mask[self.res_index]
        new_atom_idx = -np.ones(len(self.coords), dtype=int)
        new_atom_idx[atom_keep] = np.arange(atom_keep.sum())
        ca = self.ca_index[keep_res]
        return Structure(
            coords=self.coords[atom_keep], element=self.element[atom_keep],
            atom_name=self.atom_name[atom_keep], resname=self.resname[atom_keep],
            res_index=remap[self.res_index[atom_keep]], resnum=self.resnum[keep_res],
            res_names=self.res_names[keep_res], plddt=self.plddt[keep_res],
            ca_index=np.where(ca >= 0, new_atom_idx[np.maximum(ca, 0)], -1),
        )


def read_pdb(path: Path) -> Structure:
    """Parse ATOM records of the first model and first chain (AlphaFold models have one of each)."""
    coords, element, atom_name, resname, res_keys, bfac = [], [], [], [], [], []
    chain0 = None
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec == "ENDMDL":
                break
            if rec != "ATOM  ":
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            chain = line[21]
            if chain0 is None:
                chain0 = chain
            elif chain != chain0:
                break
            el = line[76:78].strip().upper() or line[12:16].strip()[0]
            if el == "H":
                continue
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            element.append(el)
            atom_name.append(line[12:16].strip())
            resname.append(line[17:20].strip())
            res_keys.append((int(line[22:26]), line[26]))
            bfac.append(float(line[60:66]))
    if not coords:
        raise ValueError(f"no ATOM records in {path}")
    keys, first_idx, inverse = np.unique(np.array([f"{k[0]:>8}{k[1]}" for k in res_keys]), return_index=True,
                                         return_inverse=True)
    # keep residues in file order
    order = np.argsort(first_idx)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    res_index = rank[inverse]
    first_sorted = first_idx[order]
    resname_arr = np.array(resname)
    atom_name_arr = np.array(atom_name)
    bfac_arr = np.array(bfac)
    n_res = len(order)
    ca_index = -np.ones(n_res, dtype=int)
    is_ca = atom_name_arr == "CA"
    ca_index[res_index[is_ca]] = np.flatnonzero(is_ca)
    plddt = np.where(ca_index >= 0, bfac_arr[np.maximum(ca_index, 0)], bfac_arr[first_sorted])
    return Structure(
        coords=np.asarray(coords, dtype=float), element=np.array(element), atom_name=atom_name_arr,
        resname=resname_arr, res_index=res_index,
        resnum=np.array([res_keys[i][0] for i in first_sorted]), res_names=resname_arr[first_sorted],
        plddt=plddt, ca_index=ca_index,
    )
