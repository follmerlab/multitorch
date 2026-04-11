"""
Track C3c tests for ``scale_atomic_params`` (Slater/SOC scaling pipeline).

What this validates
-------------------
1. Identity at scale=1.0 — the scaled tensors equal the parsed floats
   element-wise (the C3f parity test depends on this exact equality).
2. Linearity — multiplying ``slater_scale`` by 0.8 multiplies every
   Fk/Gk by 0.8, with no cross-talk into ζ.
3. Independence — ``slater_scale`` only affects Fk/Gk and ``soc_scale``
   only affects ζ.
4. Output dtype/shape — all dict values are 0-d ``DTYPE`` torch tensors.
5. Autograd through ``slater_scale`` — backward through a downstream
   sum lands on the leaf with finite, nonzero gradient that matches the
   analytical reference (the sum of every Fk/Gk).
6. Autograd through ``soc_scale`` — same contract for ζ.
7. ``zeta_method='rvi'`` selects the R*VI column instead of Blume-Watson.
8. Invalid ``zeta_method`` raises ``ValueError``.
9. ``ScaledAtomicParams`` mirror methods (``by_nconf``, ``ground``,
   ``excited``) work and the convenience accessors are order-independent
   and method-aware.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.atomic.parameter_fixtures import read_rcn31_out_params
from multitorch.atomic.scaled_params import (
    ScaledAtomicParams,
    ScaledConfigParams,
    scale_atomic_params,
)

REFDATA = Path(__file__).parent.parent / "reference_data"
NID8_RCN31 = REFDATA / "nid8" / "nid8.rcn31_out"


# ─────────────────────────────────────────────────────────────
# Module-level fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def nid8_params():
    return read_rcn31_out_params(NID8_RCN31)


@pytest.fixture(scope="module")
def nid8_scaled_identity(nid8_params):
    return scale_atomic_params(nid8_params, slater_scale=1.0, soc_scale=1.0)


# ─────────────────────────────────────────────────────────────
# Output type and structure
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_returns_scaled_atomic_params(nid8_scaled_identity):
    assert isinstance(nid8_scaled_identity, ScaledAtomicParams)


@pytest.mark.phase4
def test_two_configs_preserved(nid8_params, nid8_scaled_identity):
    assert len(nid8_scaled_identity.configs) == len(nid8_params.configs)
    for sc, pc in zip(nid8_scaled_identity.configs, nid8_params.configs):
        assert isinstance(sc, ScaledConfigParams)
        assert sc.label == pc.label
        assert sc.nconf == pc.nconf


@pytest.mark.phase4
def test_ground_excited_accessors(nid8_scaled_identity):
    assert nid8_scaled_identity.by_nconf(1) is nid8_scaled_identity.ground
    assert nid8_scaled_identity.by_nconf(2) is nid8_scaled_identity.excited


@pytest.mark.phase4
def test_unknown_nconf_raises(nid8_scaled_identity):
    with pytest.raises(KeyError):
        nid8_scaled_identity.by_nconf(99)


@pytest.mark.phase4
def test_all_fk_are_dtype_tensors(nid8_scaled_identity):
    for cfg in nid8_scaled_identity.configs:
        for key, val in cfg.fk.items():
            assert isinstance(val, torch.Tensor), f"{cfg.label} F^{key}"
            assert val.dtype == DTYPE
            assert val.ndim == 0


@pytest.mark.phase4
def test_all_gk_are_dtype_tensors(nid8_scaled_identity):
    for cfg in nid8_scaled_identity.configs:
        for key, val in cfg.gk.items():
            assert isinstance(val, torch.Tensor)
            assert val.dtype == DTYPE
            assert val.ndim == 0


@pytest.mark.phase4
def test_all_zeta_are_dtype_tensors(nid8_scaled_identity):
    for cfg in nid8_scaled_identity.configs:
        for shell, val in cfg.zeta.items():
            assert isinstance(val, torch.Tensor)
            assert val.dtype == DTYPE
            assert val.ndim == 0


# ─────────────────────────────────────────────────────────────
# Identity at scale=1.0
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_identity_fk(nid8_params, nid8_scaled_identity):
    for cfg_p, cfg_s in zip(nid8_params.configs, nid8_scaled_identity.configs):
        for key, val in cfg_p.fk.items():
            assert cfg_s.fk[key].item() == pytest.approx(val, abs=1e-12)


@pytest.mark.phase4
def test_identity_gk(nid8_params, nid8_scaled_identity):
    for cfg_p, cfg_s in zip(nid8_params.configs, nid8_scaled_identity.configs):
        for key, val in cfg_p.gk.items():
            assert cfg_s.gk[key].item() == pytest.approx(val, abs=1e-12)


@pytest.mark.phase4
def test_identity_zeta_default_is_blume_watson(nid8_params, nid8_scaled_identity):
    """Default zeta_method='blume_watson' must use ConfigParams.zeta_bw."""
    for cfg_p, cfg_s in zip(nid8_params.configs, nid8_scaled_identity.configs):
        for shell, val in cfg_p.zeta_bw.items():
            assert cfg_s.zeta[shell].item() == pytest.approx(val, abs=1e-12)


# ─────────────────────────────────────────────────────────────
# Linearity
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
@pytest.mark.parametrize("alpha", [0.8, 0.85, 0.5, 1.2])
def test_slater_scale_linearity(nid8_params, alpha):
    scaled = scale_atomic_params(nid8_params, slater_scale=alpha, soc_scale=1.0)
    for cfg_p, cfg_s in zip(nid8_params.configs, scaled.configs):
        for key, val in cfg_p.fk.items():
            assert cfg_s.fk[key].item() == pytest.approx(alpha * val, abs=1e-12)
        for key, val in cfg_p.gk.items():
            assert cfg_s.gk[key].item() == pytest.approx(alpha * val, abs=1e-12)


@pytest.mark.phase4
@pytest.mark.parametrize("beta", [0.8, 0.95, 1.0, 1.1])
def test_soc_scale_linearity(nid8_params, beta):
    scaled = scale_atomic_params(nid8_params, slater_scale=1.0, soc_scale=beta)
    for cfg_p, cfg_s in zip(nid8_params.configs, scaled.configs):
        for shell, val in cfg_p.zeta_bw.items():
            assert cfg_s.zeta[shell].item() == pytest.approx(beta * val, abs=1e-12)


@pytest.mark.phase4
def test_slater_scale_does_not_touch_zeta(nid8_params):
    """Confirm slater_scale only affects Fk/Gk."""
    scaled = scale_atomic_params(nid8_params, slater_scale=0.5, soc_scale=1.0)
    for cfg_p, cfg_s in zip(nid8_params.configs, scaled.configs):
        for shell, val in cfg_p.zeta_bw.items():
            assert cfg_s.zeta[shell].item() == pytest.approx(val, abs=1e-12)


@pytest.mark.phase4
def test_soc_scale_does_not_touch_fk_gk(nid8_params):
    """Confirm soc_scale only affects ζ."""
    scaled = scale_atomic_params(nid8_params, slater_scale=1.0, soc_scale=0.5)
    for cfg_p, cfg_s in zip(nid8_params.configs, scaled.configs):
        for key, val in cfg_p.fk.items():
            assert cfg_s.fk[key].item() == pytest.approx(val, abs=1e-12)
        for key, val in cfg_p.gk.items():
            assert cfg_s.gk[key].item() == pytest.approx(val, abs=1e-12)


# ─────────────────────────────────────────────────────────────
# zeta_method dispatch
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_zeta_method_rvi_uses_rvi_column(nid8_params):
    scaled = scale_atomic_params(nid8_params, zeta_method="rvi")
    for cfg_p, cfg_s in zip(nid8_params.configs, scaled.configs):
        for shell, val in cfg_p.zeta_rvi.items():
            assert cfg_s.zeta[shell].item() == pytest.approx(val, abs=1e-12)


@pytest.mark.phase4
def test_zeta_method_blume_watson_differs_from_rvi(nid8_params):
    """The two columns differ for at least 2P (16% gap on 3d for nid8)."""
    bw = scale_atomic_params(nid8_params, zeta_method="blume_watson")
    rvi = scale_atomic_params(nid8_params, zeta_method="rvi")
    assert bw.ground.zeta["2P"].item() != rvi.ground.zeta["2P"].item()


@pytest.mark.phase4
def test_invalid_zeta_method_raises(nid8_params):
    with pytest.raises(ValueError, match="zeta_method"):
        scale_atomic_params(nid8_params, zeta_method="bogus")


# ─────────────────────────────────────────────────────────────
# Convenience accessors on the scaled dataclass
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_f_accessor_order_independent(nid8_scaled_identity):
    a = nid8_scaled_identity.ground.f("2P", "3D", 2)
    b = nid8_scaled_identity.ground.f("3D", "2P", 2)
    assert a.item() == b.item()


@pytest.mark.phase4
def test_g_accessor_order_independent(nid8_scaled_identity):
    a = nid8_scaled_identity.ground.g("2P", "3D", 1)
    b = nid8_scaled_identity.ground.g("3D", "2P", 1)
    assert a.item() == b.item()


@pytest.mark.phase4
def test_z_accessor(nid8_scaled_identity, nid8_params):
    val = nid8_scaled_identity.ground.z("2P")
    assert val.item() == pytest.approx(nid8_params.ground.zeta_bw["2P"], abs=1e-12)


@pytest.mark.phase4
def test_f_unknown_raises(nid8_scaled_identity):
    with pytest.raises(KeyError):
        nid8_scaled_identity.ground.f("4F", "5G", 2)


@pytest.mark.phase4
def test_z_unknown_raises(nid8_scaled_identity):
    with pytest.raises(KeyError):
        nid8_scaled_identity.ground.z("4F")


# ─────────────────────────────────────────────────────────────
# Autograd: the whole reason this module exists
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_slater_scale_gradient(nid8_params):
    """Backward through a downstream sum must land on slater_scale.

    The analytical reference is sum_(all Fk and Gk in all configs).
    """
    slater_scale = torch.tensor(0.85, dtype=DTYPE, requires_grad=True)
    scaled = scale_atomic_params(
        nid8_params, slater_scale=slater_scale, soc_scale=1.0
    )

    # Sum every Fk and Gk across both configurations.
    loss = sum(
        v.sum() for cfg in scaled.configs for v in cfg.fk.values()
    ) + sum(
        v.sum() for cfg in scaled.configs for v in cfg.gk.values()
    )
    loss.backward()

    # Analytical reference: d/dα (α * Σ Fk + α * Σ Gk) = Σ Fk + Σ Gk.
    expected = sum(
        v for cfg in nid8_params.configs for v in cfg.fk.values()
    ) + sum(
        v for cfg in nid8_params.configs for v in cfg.gk.values()
    )

    assert slater_scale.grad is not None
    assert torch.isfinite(slater_scale.grad)
    assert slater_scale.grad.item() == pytest.approx(expected, abs=1e-10)


@pytest.mark.phase4
def test_soc_scale_gradient(nid8_params):
    """Backward through a downstream ζ sum must land on soc_scale."""
    soc_scale = torch.tensor(0.95, dtype=DTYPE, requires_grad=True)
    scaled = scale_atomic_params(
        nid8_params, slater_scale=1.0, soc_scale=soc_scale
    )

    loss = sum(v.sum() for cfg in scaled.configs for v in cfg.zeta.values())
    loss.backward()

    expected = sum(
        v for cfg in nid8_params.configs for v in cfg.zeta_bw.values()
    )

    assert soc_scale.grad is not None
    assert torch.isfinite(soc_scale.grad)
    assert soc_scale.grad.item() == pytest.approx(expected, abs=1e-10)


@pytest.mark.phase4
def test_slater_grad_does_not_leak_into_soc(nid8_params):
    """soc_scale.grad must remain None when only slater_scale carries the loss."""
    slater_scale = torch.tensor(0.8, dtype=DTYPE, requires_grad=True)
    soc_scale = torch.tensor(0.9, dtype=DTYPE, requires_grad=True)
    scaled = scale_atomic_params(
        nid8_params, slater_scale=slater_scale, soc_scale=soc_scale
    )
    loss = sum(v.sum() for cfg in scaled.configs for v in cfg.fk.values())
    loss.backward()
    assert slater_scale.grad is not None
    assert soc_scale.grad is None


@pytest.mark.phase4
def test_soc_grad_does_not_leak_into_slater(nid8_params):
    """slater_scale.grad must remain None when only soc_scale carries the loss."""
    slater_scale = torch.tensor(0.8, dtype=DTYPE, requires_grad=True)
    soc_scale = torch.tensor(0.9, dtype=DTYPE, requires_grad=True)
    scaled = scale_atomic_params(
        nid8_params, slater_scale=slater_scale, soc_scale=soc_scale
    )
    loss = sum(v.sum() for cfg in scaled.configs for v in cfg.zeta.values())
    loss.backward()
    assert soc_scale.grad is not None
    assert slater_scale.grad is None


@pytest.mark.phase4
def test_python_float_inputs_make_constant_tensors(nid8_params):
    """Python float scales must produce non-leaf constant outputs (no graph)."""
    scaled = scale_atomic_params(nid8_params, slater_scale=0.8, soc_scale=0.9)
    val = scaled.ground.f("3D", "3D", 2)
    assert not val.requires_grad


@pytest.mark.phase4
def test_dtype_promotion_from_float32_input(nid8_params):
    """A float32 tensor scale must be cast to DTYPE without graph break."""
    slater_scale = torch.tensor(0.8, dtype=torch.float32, requires_grad=True)
    scaled = scale_atomic_params(nid8_params, slater_scale=slater_scale)
    val = scaled.ground.f("3D", "3D", 2)
    assert val.dtype == DTYPE
    # Backward must still reach the float32 leaf.
    val.backward()
    assert slater_scale.grad is not None
    assert torch.isfinite(slater_scale.grad)
