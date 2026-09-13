"""Build the compact oh8 Fortran oracle bundled under ``fortran_ops/``.

Source: pyctm + ttmult reruns of the eight Oh fixtures on exxa at Slater
reduction 0.8 and 1.0, ``/data/ahf/multitorch/fixtures/oracle_oh8/<name>_s<red>/``
(``<name>.rcg`` = ttrcg input, ``<name>.m14`` = ttrcg RME store).

Writes
  fortran_ops/oh8_rcg/<name>_s<red>.rcg     ttrcg input decks (all parameters, 3 decimals)
  fortran_ops/oh8_s1.0_hamiltonian.npz      eigenvalues of every HAMILTONIAN block of
                                            sections 2 and 3 of the s1.0 stores, keyed
                                            "<name>/<section>/<GROUND|EXCITE>/<J>"

Usage: python tests/tools/build_oh8_oracle.py <local copy of oracle_oh8>
"""
import shutil
import sys
from pathlib import Path

import numpy as np

from multitorch.hamiltonian.build_cowan import j_value, read_cowan_metadata
from multitorch.io.read_rme import read_cowan_store

NAMES = ["ti4_d0_oh", "v3_d2_oh", "cr3_d3_oh", "mn2_d5_oh",
         "fe3_d5_oh", "fe2_d6_oh", "co2_d7_oh", "ni2_d8_oh"]
OUT = Path(__file__).resolve().parents[1] / "reference_data" / "fortran_ops"


def main(src: Path) -> None:
    (OUT / "oh8_rcg").mkdir(parents=True, exist_ok=True)
    eigs = {}
    for name in NAMES:
        for red in ("0.8", "1.0"):
            shutil.copyfile(src / f"{name}_s{red}" / f"{name}.rcg", OUT / "oh8_rcg" / f"{name}_s{red}.rcg")
        store = src / f"{name}_s1.0" / f"{name}.m14"
        mats, meta = read_cowan_store(store), read_cowan_metadata(store)
        for s in (2, 3):
            for m, M in zip(meta[s], mats[s]):
                if m.operator == "HAMILTONIAN":
                    eigs[f"{name}/{s}/{m.block_type}/{j_value(m.bra_sym)}"] = np.linalg.eigvalsh(M.numpy())
    np.savez_compressed(OUT / "oh8_s1.0_hamiltonian.npz", **eigs)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
