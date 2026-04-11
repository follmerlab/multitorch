"""
Slater / spin-orbit scaling pipeline for the Track C Phase 5 builder.

Why this module exists
----------------------
Cowan's empirical practice is to multiply the bare HFS Slater Fk/Gk
integrals by an empirical reduction factor (typically 0.8 for d-d, 0.85
for p-d) before they enter the multiplet Hamiltonian, and to similarly
re-scale the spin-orbit ζ. These scaling factors are *the* gradient
entry points for the Track C autograd story — they are not baked into
the angular machinery (``compute_*_blocks``) and they are not baked into
the file parser (``read_rcn31_out_params``). They live here because:

  1. The atomic-parameter parser yields plain Python floats, which are
     not autograd-traceable.
  2. The COWAN store builder (C3e) needs torch tensors with the scale
     factors *already applied*, so the angular block can be multiplied
     by a single scalar that already carries the gradient graph.
  3. Exposing ``slater_scale`` and ``soc_scale`` as separate function
     arguments to ``calcXAS`` (in C4) means users can compute
     ``torch.autograd.grad(loss, slater_scale)`` directly.

Contract
--------
``scale_atomic_params(params, slater_scale, soc_scale, ...)`` consumes
an :class:`AtomicParams` (parsed from ``.rcn31_out``) and returns a
:class:`ScaledAtomicParams` whose Fk/Gk/ζ values are torch tensors.

If ``slater_scale`` is a Python float, the returned tensors are leaf
constants (no graph). If it is a torch scalar tensor with
``requires_grad=True``, every Fk/Gk in the output participates in the
autograd graph through that scalar — backward through the resulting
COWAN store will land on ``slater_scale``.

The same applies independently to ``soc_scale`` for ζ.

For the Track C C3f parity test, set both scales to 1.0; the
``.rcn31_out`` values are the ab-initio HFS integrals that the
reference COWAN store was built from, so any scaling would break the
element-wise comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch

from multitorch._constants import DTYPE
from multitorch.atomic.parameter_fixtures import AtomicParams, ConfigParams

ScalarLike = Union[float, int, torch.Tensor]


# ─────────────────────────────────────────────────────────────
# Output dataclasses: torch-tensor mirrors of the parser dataclasses
# ─────────────────────────────────────────────────────────────


@dataclass
class ScaledConfigParams:
    """One configuration's atomic parameters as torch tensors.

    All Fk/Gk/ζ values are 0-dim torch tensors (``DTYPE`` = float64).
    They may carry an autograd graph reaching back to ``slater_scale``
    or ``soc_scale``, depending on whether those were torch scalars
    with ``requires_grad=True``.
    """

    label: str
    nconf: int
    fk: Dict[Tuple[str, str, int], torch.Tensor] = field(default_factory=dict)
    gk: Dict[Tuple[str, str, int], torch.Tensor] = field(default_factory=dict)
    zeta: Dict[str, torch.Tensor] = field(default_factory=dict)

    def f(self, a: str, b: str, k: int) -> torch.Tensor:
        a, b = a.upper(), b.upper()
        if (a, b, k) in self.fk:
            return self.fk[(a, b, k)]
        if (b, a, k) in self.fk:
            return self.fk[(b, a, k)]
        raise KeyError(f"F^{k}({a},{b}) not found in {self.label}")

    def g(self, a: str, b: str, k: int) -> torch.Tensor:
        a, b = a.upper(), b.upper()
        if (a, b, k) in self.gk:
            return self.gk[(a, b, k)]
        if (b, a, k) in self.gk:
            return self.gk[(b, a, k)]
        raise KeyError(f"G^{k}({a},{b}) not found in {self.label}")

    def z(self, shell: str) -> torch.Tensor:
        shell = shell.upper()
        if shell not in self.zeta:
            raise KeyError(f"ζ({shell}) not found in {self.label}")
        return self.zeta[shell]


@dataclass
class ScaledAtomicParams:
    """All scaled atomic parameters from a single .rcn31_out file."""

    configs: List[ScaledConfigParams] = field(default_factory=list)

    def by_nconf(self, nconf: int) -> ScaledConfigParams:
        for c in self.configs:
            if c.nconf == nconf:
                return c
        raise KeyError(
            f"NCONF={nconf} not found "
            f"(have {[c.nconf for c in self.configs]})"
        )

    @property
    def ground(self) -> ScaledConfigParams:
        return self.by_nconf(1)

    @property
    def excited(self) -> ScaledConfigParams:
        return self.by_nconf(2)


# ─────────────────────────────────────────────────────────────
# Scaling driver
# ─────────────────────────────────────────────────────────────


def _as_dtype_tensor(x: ScalarLike) -> torch.Tensor:
    """Coerce a scalar to a 0-d ``DTYPE`` tensor without breaking autograd.

    If ``x`` is already a torch tensor, return it as-is (so its
    ``requires_grad`` flag and any in-graph history are preserved). If
    it's a Python number, wrap it as a constant.
    """
    if isinstance(x, torch.Tensor):
        # Cast dtype if needed but keep the autograd connection.
        if x.dtype != DTYPE:
            return x.to(dtype=DTYPE)
        return x
    return torch.as_tensor(x, dtype=DTYPE)


def _scale_one_config(
    cfg: ConfigParams,
    slater_scale: torch.Tensor,
    soc_scale: torch.Tensor,
    zeta_method: str,
) -> ScaledConfigParams:
    fk_scaled: Dict[Tuple[str, str, int], torch.Tensor] = {}
    for key, val in cfg.fk.items():
        fk_scaled[key] = slater_scale * val

    gk_scaled: Dict[Tuple[str, str, int], torch.Tensor] = {}
    for key, val in cfg.gk.items():
        gk_scaled[key] = slater_scale * val

    if zeta_method == "blume_watson":
        zeta_src = cfg.zeta_bw
    elif zeta_method == "rvi":
        zeta_src = cfg.zeta_rvi
    else:
        raise ValueError(
            f"zeta_method must be 'blume_watson' or 'rvi', got {zeta_method!r}"
        )

    zeta_scaled: Dict[str, torch.Tensor] = {}
    for shell, val in zeta_src.items():
        zeta_scaled[shell] = soc_scale * val

    return ScaledConfigParams(
        label=cfg.label,
        nconf=cfg.nconf,
        fk=fk_scaled,
        gk=gk_scaled,
        zeta=zeta_scaled,
    )


def scale_atomic_params(
    params: AtomicParams,
    slater_scale: ScalarLike = 1.0,
    soc_scale: ScalarLike = 1.0,
    *,
    zeta_method: str = "blume_watson",
) -> ScaledAtomicParams:
    """Multiply Fk/Gk by ``slater_scale`` and ζ by ``soc_scale``.

    Parameters
    ----------
    params : AtomicParams
        Parsed atomic parameters from :func:`read_rcn31_out_params`.
    slater_scale : float or torch.Tensor, default 1.0
        Empirical reduction factor applied to every Fk and Gk integral.
        If a torch tensor with ``requires_grad=True``, the entire
        scaled output carries an autograd graph back to this tensor.
    soc_scale : float or torch.Tensor, default 1.0
        Empirical reduction factor applied to every spin-orbit ζ.
        Same autograd semantics as ``slater_scale``.
    zeta_method : {'blume_watson', 'rvi'}
        Which ζ column from the .rcn31_out file to use.
        ``'blume_watson'`` is the default and the recommended physics.

    Returns
    -------
    ScaledAtomicParams
        Mirror of ``params`` with all values as 0-dim torch tensors.

    Notes
    -----
    For the C3f parity test against a parsed `.rme_rcg` fixture, use
    ``slater_scale=1.0`` and ``soc_scale=1.0`` — the COWAN store in the
    fixture was built from these exact ab-initio rcn31 values without
    further scaling.
    """
    slater_t = _as_dtype_tensor(slater_scale)
    soc_t = _as_dtype_tensor(soc_scale)

    scaled_configs = [
        _scale_one_config(c, slater_t, soc_t, zeta_method)
        for c in params.configs
    ]
    return ScaledAtomicParams(configs=scaled_configs)
