"""
Parameter-linear HAMILTONIAN blocks: the one contraction both store builders use.

Every HAMILTONIAN block of a COWAN store is linear in the atomic parameters of
its configuration (energies in eV)::

    H(J) = E_av·sqrt(2J+1)·I + Σ_i p_i·O_i(J)

with parameter-free operators O_i (F^k, G^k, ζ). A :class:`ConfigDecomposition`
holds the operators of one configuration together with an *anchor*: a block set
``H_anchor`` and the parameter values ``p_anchor`` it corresponds to. Blocks at
new parameters are built as::

    H(J) = H_anchor(J) + Σ_i (p_i − p_anchor_i)·O_i(J)

* Fixture path (``hamiltonian/build_cowan.py``): the anchor is the Fortran block
  itself and ``p_anchor`` the exact least-squares fit, so the fixture's own
  reduction returns the store unchanged.
* From-scratch path (``angular/rac_generator.py``): the anchor is zero with all
  ``p_anchor = 0`` (E_av = 0), so ``H = Σ p_i O_i``.

The parameter values are ``p_i = reference_i · slater / reference_slater``
(F^k, G^k) or ``reference_i · soc / reference_soc`` (ζ): ``reference`` holds the
values at a known reduction (the fixture fit at 0.8, or the unreduced HFS values
at 1.0), unless an explicit per-parameter override is given. Dividing the
leaf by the reference reduction (rather than storing reference/0.8) keeps the
fixture store bit-exact at its own reduction. ``slater``, ``soc``
and overrides may be tensors with ``requires_grad=True``; with leading batch
dimensions they produce batched blocks ``(..., dim, dim)``.

Parameter names: operators are keyed by shell index (``F2_11``, ``F2_12``,
``G1_12``, ``zeta_2``; see :mod:`multitorch.angular.cowan_operators`).
:func:`physical_name` gives the shell-letter alias (``F2dd``, ``F2pd``,
``G1pd``, ``zeta_d``) used for overrides when it is unambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch

from multitorch._constants import DTYPE

Shell = Tuple[int, int]  # (l, n)
Scalar = Union[float, torch.Tensor]

_LETTER = "spdfghi"


def is_soc(name: str) -> bool:
    return name.startswith("zeta")


def physical_name(name: str, shells: Tuple[Shell, ...]) -> str:
    """Shell-letter alias of an operator name: ``F2_12`` on ((1,5),(2,9)) → ``F2pd``."""
    if is_soc(name):
        return f"zeta_{_LETTER[shells[int(name.split('_')[1]) - 1][0]]}"
    head, idx = name.split("_")
    return head + "".join(sorted((_LETTER[shells[int(c) - 1][0]] for c in idx), key=_LETTER.index))


@dataclass
class ConfigDecomposition:
    """One configuration's HAMILTONIAN blocks as anchor + parameter-free operators.

    ``block_index[J]`` is the position of the J block in store section
    ``section``. ``anchor[J] = E_av·sqrt(2J+1)·I + Σ anchor_params_i·operators_i[J]``
    to ``max_residual`` (relative, elementwise). ``reference`` holds the
    parameter values at reductions (``reference_slater``, ``reference_soc``).
    """

    section: int
    block_type: str
    shells: Tuple[Shell, ...]
    block_index: Dict[float, int]
    e_av: float
    anchor_params: Dict[str, float]
    reference: Dict[str, float]
    operators: Dict[str, Dict[float, torch.Tensor]]
    anchor: Dict[float, torch.Tensor]
    reference_slater: float = 1.0
    reference_soc: float = 1.0
    max_residual: float = 0.0
    label: Optional[str] = None
    # total (S, L) of every basis row of each J block, in store order
    states: Dict[float, List[Tuple[float, float]]] = field(default_factory=dict)

    # ── fixture-path vocabulary (S1a) ──
    @property
    def params(self) -> Dict[str, float]:
        return self.anchor_params

    @property
    def fixture(self) -> Dict[float, torch.Tensor]:
        return self.anchor

    def _part(self, soc: bool) -> Dict[float, torch.Tensor]:
        out = {}
        for J, H in self.anchor.items():
            part = torch.zeros_like(H)
            for n, v in self.anchor_params.items():
                if is_soc(n) == soc:
                    part = part + v * self.operators[n][J]
            out[J] = part
        return out

    @property
    def slater_part(self) -> Dict[float, torch.Tensor]:
        """Σ F^k O_F + Σ G^k O_G at the anchor."""
        return self._part(soc=False)

    @property
    def soc_part(self) -> Dict[float, torch.Tensor]:
        """Σ ζ_i O_ζi at the anchor."""
        return self._part(soc=True)

    def aliases(self) -> Dict[str, str]:
        """Unambiguous shell-letter aliases → operator names."""
        by_alias: Dict[str, List[str]] = {}
        for n in self.operators:
            by_alias.setdefault(physical_name(n, self.shells), []).append(n)
        return {a: ns[0] for a, ns in by_alias.items() if len(ns) == 1}

    def parameter_values(
        self,
        slater: Scalar,
        soc: Scalar,
        overrides: Optional[Mapping[str, Scalar]] = None,
    ) -> Dict[str, Scalar]:
        """p_i = reference_i · (slater | soc) / reference reduction, replaced by ``overrides`` (eV, absolute)."""
        a = slater / self.reference_slater
        b = soc / self.reference_soc
        values: Dict[str, Scalar] = {
            n: self.reference.get(n, 0.0) * (b if is_soc(n) else a)
            for n in self.operators
        }
        if overrides:
            aliases = self.aliases()
            for key, v in overrides.items():
                name = key if key in self.operators else aliases.get(key)
                if name is None:
                    raise KeyError(
                        f"unknown atomic parameter {key!r} for configuration "
                        f"{self.label or (self.section, self.block_type)} {self.shells}; "
                        f"known: {sorted(set(self.operators) | set(aliases))}"
                    )
                values[name] = v
        return values

    def blocks(
        self,
        values: Mapping[str, Scalar],
        device=None,
    ) -> Dict[float, torch.Tensor]:
        """H(J) = anchor(J) + Σ (p_i − anchor_i)·O_i(J); tensor p_i may carry batch dims."""
        out: Dict[float, torch.Tensor] = {}
        for J, H0 in self.anchor.items():
            H = H0.to(device=device)
            for n, v in values.items():
                d = v - self.anchor_params.get(n, 0.0)
                O = self.operators[n][J].to(device=device)
                if isinstance(d, torch.Tensor):
                    d = d.to(dtype=DTYPE, device=device)
                    H = H + d.reshape(d.shape + (1, 1)) * O
                elif d != 0.0:
                    H = H + d * O
            out[J] = H
        return out


@dataclass
class HamiltonianDecomposition:
    configs: List[ConfigDecomposition]
    slater_reduction: float = 1.0
    soc_reduction: float = 1.0

    def config(self, section: int, block_type: str) -> ConfigDecomposition:
        for c in self.configs:
            if c.section == section and c.block_type == block_type:
                return c
        raise KeyError((section, block_type))

    def by_label(self, label: str) -> ConfigDecomposition:
        for c in self.configs:
            if c.label == label:
                return c
        raise KeyError(f"no configuration labelled {label!r}; have {[c.label for c in self.configs]}")


def zero_anchor_config(
    section: int,
    block_type: str,
    shells: Tuple[Shell, ...],
    block_index: Dict[float, int],
    operators: Mapping[str, Mapping[float, object]],
    dims: Mapping[float, int],
    reference: Optional[Mapping[str, float]] = None,
    label: Optional[str] = None,
    states: Optional[Mapping[float, List[Tuple[float, float]]]] = None,
) -> ConfigDecomposition:
    """Decomposition with zero anchor (E_av = 0): ``H = Σ p_i O_i`` (from scratch)."""
    ops = {
        n: {J: torch.as_tensor(blocks[J], dtype=DTYPE) for J in block_index}
        for n, blocks in operators.items()
    }
    return ConfigDecomposition(
        section=section, block_type=block_type, shells=tuple(shells),
        block_index=dict(block_index), e_av=0.0,
        anchor_params={n: 0.0 for n in ops},
        reference=dict(reference or {}),
        operators=ops,
        anchor={J: torch.zeros(dims[J], dims[J], dtype=DTYPE) for J in block_index},
        label=label,
        states={J: list(states[J]) for J in block_index} if states else {},
    )


def as_scale(x: Scalar, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=DTYPE, device=device)
    return torch.as_tensor(float(x), dtype=DTYPE, device=device)


def rebuild_hamiltonian_store(
    template: List[List[torch.Tensor]],
    decomposition: HamiltonianDecomposition,
    *,
    slater: Scalar,
    soc: Scalar,
    atomic: Optional[Mapping[str, Mapping[str, Scalar]]] = None,
    device=None,
) -> List[List[torch.Tensor]]:
    """Store with every decomposed HAMILTONIAN block rebuilt at (slater, soc, atomic).

    ``atomic`` maps a configuration label to per-parameter overrides (absolute
    eV, operator or alias names), e.g. ``{'ex': {'G1pd': 5.2}}``. All other
    blocks are the template tensors.
    """
    slater, soc = as_scale(slater, device), as_scale(soc, device)
    atomic = dict(atomic or {})
    known = {c.label for c in decomposition.configs}
    unknown = set(atomic) - known
    if unknown:
        raise KeyError(f"atomic overrides for unknown configurations {sorted(unknown)}; have {sorted(l for l in known if l)}")
    result = [[m if device is None else m.to(device=device) for m in sec] for sec in template]
    for cfg in decomposition.configs:
        values = cfg.parameter_values(slater, soc, atomic.get(cfg.label))
        for J, H in cfg.blocks(values, device=device).items():
            result[cfg.section][cfg.block_index[J]] = H
    return result
