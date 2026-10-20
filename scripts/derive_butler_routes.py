#!/usr/bin/env python3
"""Derive crystal-field BRANCH routes from Butler (1981)'s tables.

A one-off derivation tool, not package code. It reads ``ttmult/src/disk0`` --
the ASCII Butler tables that ``ttrac`` itself reads -- and derives, for a point
group reached by a chain SO3 -> O -> G, the set of ``BRANCH`` routes that carry
a totally-symmetric (crystal-field) operator of each rank.

Why this exists
---------------
WP-B6 assumed the branch routes and block order had to come from recompiling
``ttrac``. They do not: ``ttrac`` reads them from ``disk0``, which is plain
ASCII and already in the working tree. See
``docs/EXTERNAL_CROSS_CHECKS_2026-10-20.md`` section 4.

The gate
--------
Butler's label syntax is compressed and was not documented here, so the
derivation is **not trusted until it reproduces an answer we already know**.
``pyctm/write_RAC.py`` hard-codes the D4h and Oh routes, which came from the
Fortran lineage. ``--gate`` re-derives those from ``O=>D4`` and ``SO3=>O`` and
compares. If the gate fails, nothing else this script prints means anything.

Run::

    python3 scripts/derive_butler_routes.py --gate
    python3 scripts/derive_butler_routes.py --group d3d

Usage note: reads ``disk0`` read-only; writes nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_DISK0 = Path(__file__).resolve().parents[2] / "ttmult" / "src" / "disk0"

# Butler's internal group for each external (crystallographic) name, from
# ttrac.c:4148-4169. The chain is SO3 -> O -> <internal>; an inversion group
# ("split") is its rotation subgroup plus parity, so it uses the same tables.
CHAIN = {
    "oh": ("O",),
    "o": ("O",),
    "td": ("O",),
    "d4h": ("O", "D4"),
    "d4": ("O", "D4"),
    "c4v": ("O", "D4"),
    "d2d": ("O", "D4"),
    "d3d": ("O", "D3"),
    "d3": ("O", "D3"),
    "c3v": ("O", "D3"),
    "th": ("O", "T"),
    "t": ("O", "T"),
    # Butler tabulates no O=>D2; the orthorhombic groups descend through D4.
    # Residual 49's ladder rung (6) -- subduction composition along a chain --
    # is the check that this is the same D2h as a direct descent would give.
    "d2h": ("O", "D4", "D2"),
    "d2": ("O", "D4", "D2"),
    "c2v": ("O", "D4", "D2"),
}

# The totally symmetric irrep is Butler's "0" in every group here.
IDENTITY_IRREP = "0"


def _blocks(text: str, keyword: str) -> dict[str, str]:
    """Return {name: body} for each ``<keyword> <name> ... `` block."""
    out: dict[str, str] = {}
    pattern = rf"{keyword} (\S+)(.*?)(?=\n\s*(?:GROUPDATA|BRANCHDATA)\s|\Z)"
    for match in re.finditer(pattern, text, re.S):
        out[match.group(1)] = " ".join(match.group(2).split())
    return out


def parse_irreps(body: str) -> dict[str, int]:
    """``IRREPS 0 +1 S0 -2 1 +3 ...`` -> {label: dimension}.

    The sign marks the double-group (spinor) irreps; we keep |dim| and let the
    caller decide. Boson irreps carry ``+``, fermion irreps ``-``.
    """
    match = re.search(r"IRREPS (.*?)(?: TRIADS|$)", body)
    if not match:
        return {}
    tokens = match.group(1).split()
    return {
        tokens[i]: abs(int(tokens[i + 1]))
        for i in range(0, len(tokens) - 1, 2)
        if re.fullmatch(r"[+-]\d+", tokens[i + 1])
    }


def parse_branchings(body: str) -> dict[str, list[str]]:
    """``BRANCHINGS 0 +0 ; 2 +0 +2 ; ...`` -> {parent: [child, ...]}.

    Each child token is ``[(mult)]<phase><label>``. The ``(n)`` prefix is a
    branching multiplicity; ``<phase>`` is always an explicit ``+`` or ``-``
    and is Butler's branching-coefficient phase, *not* part of the label; the
    label itself may begin with ``-``, which marks the partner of a
    complex-conjugate pair (so ``--S0`` is phase ``-`` on irrep ``-S0``, while
    ``-S0`` is phase ``-`` on irrep ``S0``).

    Only the labels matter for a boson identity route, which is all this
    script derives; the phases are parsed and discarded here deliberately,
    since using them would need its own oracle.
    """
    match = re.search(r"BRANCHINGS (.*?) ENDBRANCHINGS", body)
    if not match:
        return {}
    out: dict[str, list[str]] = {}
    for clause in match.group(1).split(";"):
        tokens = clause.split()
        if not tokens:
            continue
        parent, children = tokens[0], []
        for token in tokens[1:]:
            tok = re.fullmatch(r"(?:\((\d+)\))?([+-])(\S+)", token)
            if not tok:
                raise ValueError(f"unparsed branching token {token!r} under {parent!r}")
            mult = int(tok.group(1) or 1)
            children.extend([tok.group(3)] * mult)
        out[parent] = children
    return out


class ButlerTables:
    def __init__(self, path: Path):
        text = path.read_text()
        self.irreps = {
            name: parse_irreps(body) for name, body in _blocks(text, "GROUPDATA").items()
        }
        self.branchings = {}
        for name, body in _blocks(text, "BRANCHDATA").items():
            if "=>" in name:
                self.branchings[name] = parse_branchings(body)

    def branch(self, parent_group: str, child_group: str) -> dict[str, list[str]]:
        key = f"{parent_group}=>{child_group}"
        if key not in self.branchings:
            raise KeyError(f"no Butler branching table for {key}")
        return self.branchings[key]


def invariant_routes(tables: ButlerTables, chain: tuple[str, ...], rank: int) -> list[tuple[str, ...]]:
    """Every path SO3(rank) -> ... -> identity irrep of the last group.

    A crystal-field operator of rank k contributes one independent parameter
    per such path: it is a totally symmetric component of the rank-k field.
    """
    frontier: list[tuple[str, ...]] = [(str(rank),)]
    groups = ("SO3",) + chain
    for parent, child in zip(groups, groups[1:]):
        table = tables.branch(parent, child)
        nxt = []
        for path in frontier:
            for sub in table.get(path[-1], []):
                nxt.append(path + (sub,))
        frontier = nxt
    return [p for p in frontier if p[-1] == IDENTITY_IRREP]


def format_route(path: tuple[str, ...]) -> str:
    """Render a path the way a ttrac deck's BRANCH line does (even parity)."""
    head, *rest = path
    return f"{head}+ > 0 " + " > ".join(f"{label}+" for label in rest)


# What pyctm/write_RAC.py hard-codes, verbatim and in its own order. This is
# the oracle: it predates this script and came from the Fortran lineage, not
# from Butler's tables.
#
# Order is part of the claim, not incidental -- residual 49 names "block order"
# as one of the things the ladder cannot settle, so the comparison below is
# against an ordered sequence and must not be sorted.
#
#   hamDict['d4h']   = BRANCH 0+ > 0 0+ > 0+        (OPER HAMILTONIAN)
#   shellDict['d4h'] = BRANCH 4+ > 0 0+ > 0+        (OPER SHELL2, 3 lines
#                      BRANCH 4+ > 0 2+ > 0+         in this order)
#                      BRANCH 2+ > 0 2+ > 0+
PYCTM_KNOWN = {
    "oh": {
        "hamiltonian": ["0+ > 0 0+"],
        "shell": ["4+ > 0 0+"],
    },
    "d4h": {
        "hamiltonian": ["0+ > 0 0+ > 0+"],
        "shell": [
            "4+ > 0 0+ > 0+",
            "4+ > 0 2+ > 0+",
            "2+ > 0 2+ > 0+",
        ],
    },
}

CF_RANKS = (4, 2)


def derived_operators(tables: ButlerTables, group: str) -> dict[str, list[str]]:
    """The routes this script derives, grouped as a ttrac deck would emit them."""
    chain = CHAIN[group]
    return {
        "hamiltonian": [format_route(p) for p in invariant_routes(tables, chain, 0)],
        "shell": [
            format_route(p)
            for rank in CF_RANKS
            for p in invariant_routes(tables, chain, rank)
        ],
    }


def run_gate(tables: ButlerTables, verbose: bool = True) -> bool:
    if verbose:
        print("GATE -- re-deriving routes we already know, from Butler's tables")
        print("        oracle: pyctm/write_RAC.py, ordered\n")
    ok = True
    for group, expected in PYCTM_KNOWN.items():
        got = derived_operators(tables, group)
        for oper in ("hamiltonian", "shell"):
            matched = got[oper] == expected[oper]
            ok &= matched
            if verbose:
                print(f"  [{'ok ' if matched else 'FAIL'}] {group:4s} {oper:12s} {got[oper]}")
                if not matched:
                    print(f"         pyctm says: {expected[oper]}")
    if verbose:
        print()
        print("GATE PASSED" if ok else "GATE FAILED -- derivation is not trusted")
    return ok


def run_selftest(tables: ButlerTables) -> bool:
    """Prove the gate can fail.

    A gate that has never been seen to fail is not a gate. Each perturbation
    below is applied to the **in-memory** tables -- ``disk0`` is read-only
    source data and is never written -- and the gate must reject it.
    """
    print("SELFTEST -- the gate must reject each injected error\n")
    import copy

    injections = {
        "O=>D4 loses E -> A1 (kills a rank-4 and the rank-2 route)":
            lambda t: t.branchings["O=>D4"].__setitem__("2", ["2"]),
        "O=>D4 gains a spurious T1 -> A1 (adds routes)":
            lambda t: t.branchings["O=>D4"].__setitem__("1", ["^0", "1", "0"]),
        "SO3=>O drops A1 from rank 4 (kills the cubic invariant)":
            lambda t: t.branchings["SO3=>O"].__setitem__("4", ["1", "2", "^1"]),
        "SO3=>O swaps the rank-4 order (order must matter)":
            lambda t: t.branchings["SO3=>O"].__setitem__("4", ["2", "0", "1", "^1"]),
    }

    all_rejected = True
    for label, inject in injections.items():
        broken = copy.deepcopy(tables)
        inject(broken)
        rejected = not run_gate(broken, verbose=False)
        all_rejected &= rejected
        print(f"  [{'ok  ' if rejected else 'MISS'}] {label}")

    print()
    if all_rejected:
        print("SELFTEST PASSED -- the gate discriminates")
    else:
        print("SELFTEST FAILED -- the gate passed an injected error and proves nothing")
    return all_rejected


def report(tables: ButlerTables, group: str) -> None:
    chain = CHAIN[group]
    print(f"\n{group} -- chain SO3 -> " + " -> ".join(chain))
    opers = derived_operators(tables, group)
    for route in opers["hamiltonian"]:
        print(f"  OPER HAMILTONIAN   BRANCH {route}")
    for route in opers["shell"]:
        print(f"  OPER SHELL         BRANCH {route}")
    print(f"  => {len(opers['shell'])} crystal-field parameters beyond the identity")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk0", type=Path, default=DEFAULT_DISK0)
    parser.add_argument("--gate", action="store_true", help="re-derive the known Oh/D4h routes")
    parser.add_argument("--selftest", action="store_true", help="prove the gate can fail")
    parser.add_argument("--group", action="append", default=[], choices=sorted(CHAIN))
    args = parser.parse_args(argv)

    if not args.disk0.exists():
        print(f"disk0 not found at {args.disk0}", file=sys.stderr)
        return 2
    tables = ButlerTables(args.disk0)

    # The gate always runs: no derived route is reported without it.
    if not run_gate(tables):
        return 1
    if args.selftest:
        print()
        if not run_selftest(tables):
            return 1
    for group in args.group:
        report(tables, group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
