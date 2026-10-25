"""
SCA-001: the broadening-mode default, and the guards that keep it honest.

Through 0.1.x every entry point defaulted to ``broaden_mode="legacy"``, which
reproduces pyctm's transcription error in the eta mixing parameter (the second
Thompson term linear instead of quadratic). The default is now ``'correct'``,
reached through ``broaden_mode=None``.

What is tested here, and why each one has teeth:

  1. ``None`` resolves to ``'correct'`` and warns; an explicit mode is silent.
     ``'legacy'`` is NOT deprecated -- the bundled `.xy` reference spectra are
     its consumer -- so it must not warn either.
  2. An unrecognised mode raises. It used to be silently treated as
     ``'correct'``, so ``mode='Legacy'`` quietly selected the other formula.
  3. Every function that takes ``broaden_mode`` defaults it to ``None``,
     discovered from the module rather than from a list here, so a seventh
     entry point added with ``="legacy"`` fails.
  4. The resolved mode reaches the lineshape. A default that resolves correctly
     and is then dropped on the floor would pass 1-3; only an end-to-end
     comparison of spectra catches that.
  5. The resolution happens before the pipeline, asserted by which of two
     errors arrives first rather than by reading the source.
"""
import inspect
import warnings

import numpy as np
import pytest
import torch

from pathlib import Path

from multitorch.spectrum.broaden import (
    BROADEN_MODES,
    DEFAULT_BROADEN_MODE,
    pseudo_voigt,
    resolve_broaden_mode,
)

REFROOT = Path(__file__).parent.parent / "reference_data"


@pytest.mark.phase1
def test_default_resolves_to_correct_and_says_so():
    """None takes the package default, and the change is not silent."""
    assert DEFAULT_BROADEN_MODE == "correct"
    with pytest.warns(FutureWarning, match="default broadening mode changed"):
        assert resolve_broaden_mode(None) == "correct"


@pytest.mark.phase1
@pytest.mark.parametrize("mode", BROADEN_MODES)
def test_explicit_mode_never_warns(mode):
    """'legacy' is a supported compatibility mode, not a deprecated one."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_broaden_mode(mode) == mode


@pytest.mark.phase1
@pytest.mark.parametrize("bad", ["Legacy", "Correct", "corect", "", "voigt"])
def test_unrecognised_mode_raises(bad):
    """Used to fall through to the correct formula silently."""
    with pytest.raises(ValueError, match="not one of"):
        resolve_broaden_mode(bad)
    x = torch.linspace(-2.0, 2.0, 64, dtype=torch.float64)
    x0 = torch.zeros(1, dtype=torch.float64)
    amp = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="not one of"):
        pseudo_voigt(x, x0, amp, fwhm_g=0.2, fwhm_l=0.2, mode=bad)


@pytest.mark.phase1
def test_every_broaden_mode_parameter_defaults_to_none():
    """
    Discovered from the module, not listed here: a new entry point that
    hardcodes a mode fails this, and so does one that drops the parameter.
    """
    from multitorch.api import calc

    found = {}
    for name, fn in vars(calc).items():
        if not inspect.isfunction(fn) or fn.__module__ != calc.__name__:
            continue
        param = inspect.signature(fn).parameters.get("broaden_mode")
        if param is not None:
            found[name] = param.default

    # The discovery itself must not be the thing that broke: SCA-001 counted
    # six signatures, and a vacuous loop would otherwise pass.
    assert len(found) >= 6, found
    assert all(default is None for default in found.values()), found

    assert inspect.signature(pseudo_voigt).parameters["mode"].default is None


@pytest.mark.phase1
@pytest.mark.integration
def test_the_default_reaches_the_lineshape():
    """
    Resolving to 'correct' is worthless if the resolved value never arrives.
    The defaulted spectrum must be the 'correct' one bit for bit, and must
    differ from the legacy one by the measured peak-height shift.
    """
    from multitorch.api.plot import getXAS

    ban = REFROOT / "ni2_d8_oh" / "ni2_d8_oh.ban_out"
    kw = dict(T=80.0, beam_fwhm=0.2, gamma1=0.2, gamma2=0.4, med_energy=25.0,
              xmin=0.0, xmax=60.0, nbins=1200)

    with pytest.warns(FutureWarning):
        x_def, y_def = getXAS(str(ban), **kw)
    x_cor, y_cor = getXAS(str(ban), broaden_mode="correct", **kw)
    _, y_leg = getXAS(str(ban), broaden_mode="legacy", **kw)

    assert torch.equal(x_def, x_cor)
    assert torch.equal(y_def, y_cor), "the defaulted mode is not 'correct'"

    # Measured 2026-10-25: correct lowers Ni(II)'s peak by 4.6% of the legacy
    # peak and holds the area to 5e-4. A flip that silently did nothing would
    # make this ratio 1.0.
    peak_ratio = float(y_cor.max() / y_leg.max())
    assert 0.93 < peak_ratio < 0.97, peak_ratio
    area_ratio = float(torch.trapezoid(y_cor, x_cor) / torch.trapezoid(y_leg, x_cor))
    assert area_ratio == pytest.approx(1.0, abs=2e-3), area_ratio


@pytest.mark.phase1
def test_a_bad_mode_is_rejected_before_any_work():
    """
    Which error arrives decides this, not where the call to the resolver sits
    in the source: an unsupported symmetry is rejected by
    ``_validate_from_scratch``, which runs after the mode is resolved.
    """
    from multitorch.api.calc import calcXAS_from_scratch

    with pytest.raises(ValueError, match="not one of"):
        calcXAS_from_scratch("Ni", "2+", cf={"tendq": 1.0}, sym="c2v",
                             broaden_mode="nope")
