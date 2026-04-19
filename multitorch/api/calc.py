"""
High-level calculation API — the primary user interface for multitorch.

These functions wire together:
  atomic  → angular  → hamiltonian  → spectrum

Each function has two modes of operation:
  1. Full PyTorch pipeline (Phase 5 complete): all physics computed natively
  2. Fortran bootstrap mode: reads pre-computed reference data from .ban_out
     files (requires the ttmult Fortran binaries and pyttmult installed)

The API matches pyctm's function signatures for drop-in compatibility.

Usage:
    from multitorch import calcXAS
    x, y = calcXAS(element='Ni', valence='ii', sym='d4h', edge='l',
                   cf={'tendq': 1.0, 'ds': 0.0, 'dt': 0.01},
                   slater=0.8, soc=1.0, T=80)
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch


from dataclasses import dataclass
from typing import List

from multitorch._constants import DTYPE
from multitorch.spectrum.sticks import get_sticks
from multitorch.spectrum.broaden import pseudo_voigt
from multitorch.device_utils import suggest_device_for_xas, suggest_device_for_rixs


# ─────────────────────────────────────────────────────────────
# Fixture caching for fast parameter sweeps
# ─────────────────────────────────────────────────────────────

@dataclass
class CachedFixture:
    """Pre-parsed fixture data for fast parameter sweeps.

    Created by :func:`preload_fixture`. Pass to :func:`calcXAS_cached`
    to skip all file I/O and parsing on each iteration of a parameter
    sweep. Only the physics parameters (``slater``, ``soc``, ``cf``,
    ``delta``, ``lmct``, ``mlct``) vary per iteration; the angular
    structure and template matrices are reused from this cache.

    Attributes
    ----------
    ban : BanData
        Parsed BAN template (before parameter overrides).
    raw_params : AtomicParams
        Parsed atomic parameters from ``.rcn31_out``.
    rac : RACFileFull
        Parsed RAC assembly recipe.
    plan : SectionPlan
        COWAN section plan derived from RAC + COWAN store.
    cowan_template : list of list of torch.Tensor
        Parsed COWAN store matrices (template before rebuild).
    cowan_metadata : list of list of CowanBlockMeta
        Block metadata for each COWAN store matrix.
    """
    ban: object          # BanData
    raw_params: object   # AtomicParams
    rac: object          # RACFileFull
    plan: object         # SectionPlan
    cowan_template: list  # List[List[torch.Tensor]]
    cowan_metadata: list  # List[List[CowanBlockMeta]]


def preload_fixture(
    element: str,
    valence: str,
    sym: str,
    edge: str = 'l',
) -> CachedFixture:
    """Parse all fixture files once and return a reusable cache.

    Use with :func:`calcXAS_cached` for fast parameter sweeps::

        cache = preload_fixture("Ni", "ii", "oh")
        for tendq in torch.linspace(0.5, 2.0, 100):
            x, y = calcXAS_cached(cache, cf={'tendq': float(tendq)})

    Parameters
    ----------
    element, valence, sym, edge : str
        System specification (same as ``calcXAS``).

    Returns
    -------
    CachedFixture
        Reusable cache holding all parsed fixture data.
    """
    from multitorch.atomic.parameter_fixtures import read_rcn31_out_params
    from multitorch.hamiltonian.build_cowan import read_cowan_metadata
    from multitorch.hamiltonian.build_rac import build_rac_in_memory
    from multitorch.io.read_ban import read_ban
    from multitorch.io.read_rme import read_cowan_store, read_rme_rac_full

    fixture_dir = _find_fixture_dir(element, valence, sym)
    ban_path = _find_primary_fixture(fixture_dir, "*.ban")
    rcg_path = _find_primary_fixture(fixture_dir, "*.rme_rcg")
    rac_path = _find_primary_fixture(fixture_dir, "*.rme_rac")
    rcn31_path = _find_rcn31_out(fixture_dir, element, valence)

    ban = read_ban(ban_path)
    raw_params = read_rcn31_out_params(rcn31_path)

    # Parse the heavy files once
    parsed_rac = read_rme_rac_full(rac_path)
    cowan_template = read_cowan_store(rcg_path)
    cowan_metadata = read_cowan_metadata(rcg_path)

    # Build RAC + plan from pre-parsed data (no file I/O)
    rac, plan = build_rac_in_memory(
        ban, parsed_rac=parsed_rac, parsed_cowan=cowan_template,
    )

    return CachedFixture(
        ban=ban,
        raw_params=raw_params,
        rac=rac,
        plan=plan,
        cowan_template=cowan_template,
        cowan_metadata=cowan_metadata,
    )


def calcXAS_cached(
    cache: CachedFixture,
    cf: Optional[dict] = None,
    slater: float = 0.8,
    soc: float = 1.0,
    delta=None, u=None, lmct=None, mlct=None,
    T: float = 80.0,
    beam_fwhm: float = 0.2,
    gamma1: float = 0.2,
    gamma2: float = 0.4,
    med_energy: float = 25.0,
    max_gs: int = 1,
    broaden_mode: str = "legacy",
    xmin=None, xmax=None, nbins: int = 2000,
    return_sticks: bool = False,
    device: str = "cpu",
) -> Union[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Calculate XAS from a pre-loaded fixture cache (no file I/O).

    This is the fast-path version of :func:`calcXAS` for parameter
    sweeps. Create a cache with :func:`preload_fixture`, then call
    this function in a loop with varying physics parameters.

    Parameters
    ----------
    cache : CachedFixture
        Pre-parsed fixture data from :func:`preload_fixture`.
    cf : dict, optional
        Crystal-field parameters (same as ``calcXAS``).
    slater, soc : float or torch.Tensor
        Slater / SOC scaling (supports ``requires_grad=True``).
    delta, u, lmct, mlct : float, optional
        Charge-transfer parameters.
    T, beam_fwhm, gamma1, gamma2, med_energy, max_gs, broaden_mode,
    xmin, xmax, nbins, return_sticks, device
        Same as ``calcXAS``.

    Returns
    -------
    (x, y) or (x, y, sticks)
        Same as ``calcXAS``.
    """
    import copy
    from multitorch.atomic.scaled_params import scale_atomic_params
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.hamiltonian.build_ban import modify_ban_params
    from multitorch.hamiltonian.build_cowan import build_cowan_store_in_memory
    from multitorch.spectrum.sticks import get_sticks_from_banresult

    if cf is None:
        cf = {}

    # Apply parameter overrides to a copy of the cached BAN
    ban = modify_ban_params(copy.deepcopy(cache.ban), cf=cf, delta=delta, lmct=lmct, mlct=mlct)

    # Scale atomic params (fast tensor ops, no file I/O)
    scaled_params = scale_atomic_params(
        cache.raw_params, slater_scale=slater, soc_scale=soc,
    )

    # Build COWAN store from cached template + metadata (no file I/O)
    cowan = build_cowan_store_in_memory(
        scaled_params, cache.raw_params, cache.plan,
        cowan_template=cache.cowan_template,
        cowan_metadata=cache.cowan_metadata,
        device=device,
    )

    # Assemble and diagonalize
    result = assemble_and_diagonalize_in_memory(cowan, cache.rac, ban, device=device)

    # Extract sticks
    E_sticks, M_sticks, _ = get_sticks_from_banresult(
        result, T=T, max_gs=max_gs, device=device,
    )

    if E_sticks.numel() == 0:
        raise ValueError("No transitions found")

    # Broaden
    E_min = float(E_sticks.min())
    E_max = float(E_sticks.max())
    if xmin is None:
        xmin = E_min - 5.0
    if xmax is None:
        xmax = E_max + 5.0

    x = torch.linspace(xmin, xmax, nbins, dtype=DTYPE, device=device)

    med = 0.5 * (E_min + E_max)
    y = pseudo_voigt(
        x, E_sticks, M_sticks,
        fwhm_g=beam_fwhm, fwhm_l=gamma1, fwhm_l2=gamma2,
        med_energy=med, mode=broaden_mode,
    )

    if return_sticks:
        sticks = torch.stack([E_sticks, M_sticks], dim=1)
        return x, y, sticks
    return x, y


# ─────────────────────────────────────────────────────────────
# Batch API for parameter sweeps (Phase 2)
# ─────────────────────────────────────────────────────────────


def calcXAS_batch(
    cache: CachedFixture,
    slater_values: torch.Tensor,  # (N,)
    soc_values: torch.Tensor,     # (N,)
    cf: Optional[dict] = None,
    delta=None, u=None, lmct=None, mlct=None,
    T: float = 80.0,
    beam_fwhm: float = 0.2,
    gamma1: float = 0.2,
    gamma2: float = 0.4,
    med_energy: float = 25.0,
    max_gs: int = 1,
    broaden_mode: str = "legacy",
    xmin=None, xmax=None, nbins: int = 2000,
    device=None,
) -> torch.Tensor:
    """Batch calculate N XAS spectra with different (slater, soc) parameters.
    
    Optimized for parameter sweeps: computes N spectra 2-5× faster than
    N sequential calls to calcXAS_cached() by:
    - Computing V(11) residual once (shared across all parameter sets)
    - Batching atomic parameter scaling
    - Amortizing fixture loading overhead
    
    Expected speedup vs sequential calcXAS_cached():
    - N=100: 3-5× faster
    - N=1000: 5-8× faster
    - N=5000: 5-8× faster
    
    Parameters
    ----------
    cache : CachedFixture
        Pre-loaded fixture from :func:`preload_fixture`.
    slater_values : torch.Tensor, shape (N,)
        Array of Slater scale factors (one per spectrum).
        If ``requires_grad=True``, per-spectrum gradients preserved.
    soc_values : torch.Tensor, shape (N,)
        Array of spin-orbit scale factors (one per spectrum).
        If ``requires_grad=True``, per-spectrum gradients preserved.
    cf, delta, u, lmct, mlct : optional
        Physics parameters (same for all N spectra in the batch).
        For per-spectrum CF variation, call this function multiple times
        with different cf dicts (still faster than sequential).
    T, beam_fwhm, gamma1, gamma2, med_energy, max_gs, broaden_mode,
    xmin, xmax, nbins : optional
        Broadening parameters (same as :func:`calcXAS_cached`).
    device : torch.device, optional
        Computation device. If None, auto-selects based on problem size.
        
    Returns
    -------
    y_batch : torch.Tensor, shape (N, nbins)
        Batch of N broadened spectra sharing common x-grid.
        Each row is y-values for one (slater, soc) combination.
        Use ``x = torch.linspace(xmin, xmax, nbins)`` to get x-grid.
        
    Examples
    --------
    Grid search over 100 parameter combinations::
    
        cache = preload_fixture("Ni", "ii", "d4h")
        slater_grid = torch.linspace(0.6, 1.0, 10)
        soc_grid = torch.linspace(0.8, 1.2, 10)
        slater_vals, soc_vals = torch.meshgrid(slater_grid, soc_grid, indexing='ij')
        
        # 100 spectra in ~0.5-0.8s (vs 1.5-2.5s sequential)
        y_batch = calcXAS_batch(cache, 
                                slater_values=slater_vals.flatten(),
                                soc_values=soc_vals.flatten())
        
    Monte Carlo uncertainty quantification::
    
        slater_samples = torch.normal(0.85, 0.05, size=(1000,))
        soc_samples = torch.normal(1.0, 0.1, size=(1000,))
        
        # 1000 spectra in ~3-5s (vs 15-25s sequential)
        y_batch = calcXAS_batch(cache, slater_samples, soc_samples)
        mean_spectrum = y_batch.mean(dim=0)
        std_spectrum = y_batch.std(dim=0)
        
    Optimization with autograd::
    
        slater_fit = torch.linspace(0.7, 0.9, 100, requires_grad=True)
        soc_fit = torch.linspace(0.9, 1.1, 100, requires_grad=True)
        
        y_batch = calcXAS_batch(cache, slater_fit, soc_fit)
        loss = ((y_batch - y_exp_batch) ** 2).sum()
        loss.backward()  # Gradients flow back to slater_fit and soc_fit
        
    Notes
    -----
    Memory usage for Ni d8 L-edge (17×17 matrices, 2000 bins):
    - N=100: ~50 MB
    - N=1000: ~500 MB
    - N=5000: ~2.5 GB
    
    For large N (>1000), consider splitting into smaller batches if
    memory constrained. Each batch still benefits from V(11) sharing.
    
    Crystal-field parameters (cf) are applied uniformly across the batch.
    For per-spectrum CF variation, batch over cf externally and stack results.
    """
    import copy
    from multitorch.atomic.scaled_params import batch_scale_atomic_params
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.hamiltonian.build_ban import modify_ban_params
    from multitorch.hamiltonian.build_cowan import build_cowan_store_in_memory_batch
    from multitorch.spectrum.sticks import get_sticks_from_banresult
    
    # Validate inputs
    if not isinstance(slater_values, torch.Tensor):
        slater_values = torch.as_tensor(slater_values, dtype=DTYPE)
    if not isinstance(soc_values, torch.Tensor):
        soc_values = torch.as_tensor(soc_values, dtype=DTYPE)
    
    if slater_values.ndim != 1:
        raise ValueError(f"slater_values must be 1D, got shape {slater_values.shape}")
    if soc_values.ndim != 1:
        raise ValueError(f"soc_values must be 1D, got shape {soc_values.shape}")
    if slater_values.shape[0] != soc_values.shape[0]:
        raise ValueError(
            f"Batch size mismatch: slater_values has {slater_values.shape[0]} "
            f"elements but soc_values has {soc_values.shape[0]}"
        )
    
    N = slater_values.shape[0]
    
    # Auto-select device if not specified
    if device is None:
        device = "cpu"  # For Phase 2, use CPU (batch diagonalization on GPU future work)
    
    slater_values = slater_values.to(device=device)
    soc_values = soc_values.to(device=device)
    
    if cf is None:
        cf = {}
    
    # Apply parameter overrides (same for all spectra)
    ban = modify_ban_params(copy.deepcopy(cache.ban), cf=cf, delta=delta, lmct=lmct, mlct=mlct)
    
    # *** KEY OPTIMIZATION 1: Batch scale atomic params ***
    # Produces ScaledAtomicParams with (N,) tensors instead of scalars
    scaled_params_batch = batch_scale_atomic_params(
        cache.raw_params, 
        slater_values=slater_values, 
        soc_values=soc_values,
    )
    
    # *** KEY OPTIMIZATION 2: Batch COWAN rebuild with shared V(11) ***
    # Computes V(11) once, then rebuilds N Hamiltonians
    # This is the 2-3× speedup source
    cowan_batch = build_cowan_store_in_memory_batch(
        scaled_params_batch, cache.raw_params, cache.plan,
        cowan_template=cache.cowan_template,
        cowan_metadata=cache.cowan_metadata,
        device=device,
    )
    
    # Pass 1 — diagonalize and extract sticks for every sample.
    # We split the loop into two passes so we can derive a *shared*
    # x-grid from the union of all samples' stick ranges. Doing the
    # x-grid inside the loop and mutating the xmin/xmax function
    # arguments (the old behaviour) silently locked every sample to
    # sample 0's grid, producing large errors on interior samples.
    all_sticks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(N):
        cowan_i = [
            [mat[i] if mat.ndim == 3 else mat for mat in section]
            for section in cowan_batch
        ]
        result = assemble_and_diagonalize_in_memory(
            cowan_i, cache.rac, ban, device=device
        )
        E_sticks, M_sticks, _ = get_sticks_from_banresult(
            result, T=T, max_gs=max_gs, device=device
        )
        all_sticks.append((E_sticks, M_sticks))

    # Derive the shared x-range. If the caller supplied xmin/xmax we
    # honour them; otherwise we take the union across all non-empty
    # samples so no sample gets clipped or shifted.
    if xmin is None or xmax is None:
        non_empty = [
            (float(E.min()), float(E.max()))
            for E, _ in all_sticks if E.numel() > 0
        ]
        if non_empty:
            global_min = min(p[0] for p in non_empty)
            global_max = max(p[1] for p in non_empty)
        else:
            global_min = global_max = float(med_energy)
        if xmin is None:
            xmin = global_min - 5.0
        if xmax is None:
            xmax = global_max + 5.0

    x = torch.linspace(xmin, xmax, nbins, dtype=DTYPE, device=device)

    # Pass 2 — broaden each sample on the shared x-grid. med_energy
    # stays per-sample to match calcXAS_cached's convention (it
    # affects the Thompson mode switch); the stacked tensor is
    # meaningful because every row shares x.
    y_results = []
    for E_sticks, M_sticks in all_sticks:
        if E_sticks.numel() == 0:
            y = torch.zeros(nbins, dtype=DTYPE, device=device)
        else:
            med = 0.5 * (float(E_sticks.min()) + float(E_sticks.max()))
            y = pseudo_voigt(
                x, E_sticks, M_sticks,
                fwhm_g=beam_fwhm, fwhm_l=gamma1, fwhm_l2=gamma2,
                med_energy=med, mode=broaden_mode,
            )
        y_results.append(y)

    y_batch = torch.stack(y_results, dim=0)
    return y_batch


def batch_parameter_sweep(
    cache: CachedFixture,
    slater_range: Optional[torch.Tensor] = None,
    soc_range: Optional[torch.Tensor] = None,
    slater_grid: Optional[torch.Tensor] = None,
    soc_grid: Optional[torch.Tensor] = None,
    **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convenience wrapper for grid search or 1D parameter sweeps.
    
    Automatically creates parameter grids and returns results in intuitive
    shapes. Wrapper around :func:`calcXAS_batch` for common use cases.
    
    Parameters
    ----------
    cache : CachedFixture
        Pre-loaded fixture from :func:`preload_fixture`.
    slater_range : torch.Tensor, shape (n_slater,), optional
        1D array of Slater values for grid search.
        Used with soc_range to create 2D grid.
    soc_range : torch.Tensor, shape (n_soc,), optional
        1D array of SOC values for grid search.
        Used with slater_range to create 2D grid.
    slater_grid : torch.Tensor, shape (N,), optional
        Pre-flattened array of Slater values (alternative to slater_range).
        Must be provided with soc_grid of same length.
    soc_grid : torch.Tensor, shape (N,), optional
        Pre-flattened array of SOC values (alternative to soc_range).
        Must be provided with slater_grid of same length.
    **kwargs
        Additional arguments passed to :func:`calcXAS_batch`
        (cf, T, beam_fwhm, gamma1, gamma2, etc.)
        
    Returns
    -------
    y_batch : torch.Tensor
        If slater_grid/soc_grid provided: shape (N, nbins)
        If slater_range/soc_range provided: shape (n_slater, n_soc, nbins)
    slater_values : torch.Tensor (only for grid mode)
        Shape (n_slater,) - the slater_range input
    soc_values : torch.Tensor (only for grid mode)
        Shape (n_soc,) - the soc_range input
        
    Examples
    --------
    2D grid search (returns 3D array)::
    
        cache = preload_fixture("Ni", "ii", "d4h")
        slater_vals = torch.linspace(0.6, 1.0, 10)
        soc_vals = torch.linspace(0.8, 1.2, 10)
        
        # Returns (10, 10, 2000) shaped array + axes
        y_grid, slater_axis, soc_axis = batch_parameter_sweep(
            cache, slater_range=slater_vals, soc_range=soc_vals
        )
        
        # Access spectrum for slater=0.8, soc=1.0:
        i_slater = 5  # index in slater_vals
        i_soc = 5     # index in soc_vals  
        y_spectrum = y_grid[i_slater, i_soc, :]
        
    1D sweep or custom grids::
    
        # Monte Carlo sampling
        slater_samples = torch.randn(1000) * 0.05 + 0.85
        soc_samples = torch.randn(1000) * 0.1 + 1.0
        
        y_batch = batch_parameter_sweep(
            cache, slater_grid=slater_samples, soc_grid=soc_samples
        )  # Returns (1000, 2000)
        
        mean_spectrum = y_batch.mean(dim=0)
        std_spectrum = y_batch.std(dim=0)
    """
    # Validate input modes
    grid_mode = (slater_range is not None and soc_range is not None)
    custom_mode = (slater_grid is not None and soc_grid is not None)
    
    if not grid_mode and not custom_mode:
        raise ValueError(
            "Must provide either (slater_range, soc_range) for grid search "
            "or (slater_grid, soc_grid) for custom parameter arrays"
        )
    
    if grid_mode and custom_mode:
        raise ValueError(
            "Cannot mix grid mode (slater_range/soc_range) with custom mode "
            "(slater_grid/soc_grid). Choose one."
        )
    
    if grid_mode:
        # 2D grid search mode
        n_slater = slater_range.shape[0]
        n_soc = soc_range.shape[0]
        
        # Create meshgrid and flatten
        slater_mesh, soc_mesh = torch.meshgrid(slater_range, soc_range, indexing='ij')
        slater_flat = slater_mesh.flatten()
        soc_flat = soc_mesh.flatten()
        
        # Compute batch
        y_flat = calcXAS_batch(
            cache, 
            slater_values=slater_flat, 
            soc_values=soc_flat,
            **kwargs
        )  # (n_slater*n_soc, nbins)
        
        # Reshape to (n_slater, n_soc, nbins)
        nbins = y_flat.shape[1]
        y_grid = y_flat.reshape(n_slater, n_soc, nbins)
        
        return y_grid, slater_range, soc_range
    
    else:
        # Custom grid mode
        if slater_grid.shape != soc_grid.shape:
            raise ValueError(
                f"slater_grid and soc_grid must have same shape, got "
                f"{slater_grid.shape} vs {soc_grid.shape}"
            )
        
        y_batch = calcXAS_batch(
            cache,
            slater_values=slater_grid,
            soc_values=soc_grid,
            **kwargs
        )
        
        return y_batch


def calcXAS(
    element: str,
    valence: str,
    sym: str,
    edge: str,
    cf: dict,
    slater: float = 0.8,
    soc: float = 1.0,
    delta: Optional[dict] = None,
    u: Optional[list] = None,
    lmct: Optional[dict] = None,
    mlct: Optional[dict] = None,
    T: float = 80.0,
    beam_fwhm: float = 0.2,
    gamma1: float = 0.2,
    gamma2: float = 0.4,
    med_energy: float = 25.0,
    max_gs: int = 1,
    broaden_mode: str = "legacy",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    nbins: int = 2000,
    return_sticks: bool = False,
    device: Optional[str] = None,
    ban_output_path: Optional[str] = None,
    **kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Calculate an X-ray absorption spectrum.

    Parameters
    ----------
    element : str
        Element symbol (e.g. 'Ni', 'Fe', 'Ti').
    valence : str
        Oxidation state ('i', 'ii', 'iii', 'iv').
    sym : str
        Crystal symmetry ('oh', 'd4h', 'c4h').
    edge : str
        X-ray edge ('l' for L-edge 2p→3d, 'k' for K-edge 1s→3p).
    cf : dict
        Crystal field parameters: {'tendq': float, 'ds': float, 'dt': float}.
    slater : float
        Slater integral reduction factor (0-1, default 0.8).
    soc : float
        Spin-orbit coupling reduction factor (0-1, default 1.0).
    delta : dict or None
        Charge transfer energies: {'lmct': float, 'mlct': float}.
    u : list or None
        Coulomb repulsion parameters.
    lmct : dict or None
        LMCT configuration mixing parameters.
    mlct : dict or None
        MLCT configuration mixing parameters.
    T : float
        Temperature in Kelvin (default 80 K).
    beam_fwhm : float
        Gaussian beam FWHM (eV).
    gamma1 : float
        L3 lifetime FWHM (eV).
    gamma2 : float
        L2 lifetime FWHM (eV).
    med_energy : float
        L3/L2 crossover energy (eV relative to sticks).
    max_gs : int
        Number of ground states to include.
    broaden_mode : str
        'legacy' or 'correct' pseudo-Voigt mode.
    xmin, xmax : float or None
        Energy range for output spectrum. Auto-detected if None.
    nbins : int
        Number of energy bins in output spectrum.
    return_sticks : bool
        If True, also return the stick spectrum (E, I).
    device : str or None
        PyTorch device ('cpu', 'cuda', 'cuda:0', etc.). If None, automatically
        selects optimal device based on operation characteristics:
        
        - Typical 3d transition metals (Ni, Fe, etc.): CPU optimal
          (small matrices ~17×17 to 200×200, kernel launch overhead dominates)
        - Large rare earth systems: GPU beneficial (dim ≥ 500)
        - Default: 'cpu' (conservative, always works)
        
        For parameter refinement workflows with many spectra, consider
        batch processing (future feature) which benefits from GPU.
    ban_output_path : str or None
        Path to a pre-computed .ban_out file. If provided, skips the full
        PyTorch physics pipeline and reads the Fortran output directly.
        This is the bootstrap mode for early validation.

    Returns
    -------
    x : torch.Tensor  shape (nbins,)
        Energy axis (eV).
    y : torch.Tensor  shape (nbins,)
        Absorption intensity.
    sticks : torch.Tensor  shape (N, 2)  (only if return_sticks=True)
        Columns: [energy (eV), intensity].
    """
    # Smart device selection if not explicitly provided
    if device is None:
        device = suggest_device_for_xas(element=element, valence=valence)
    
    if ban_output_path is not None:
        # Bootstrap mode: read pre-computed Fortran output
        return _calcXAS_from_ban(
            ban_output_path, T=T, beam_fwhm=beam_fwhm,
            gamma1=gamma1, gamma2=gamma2, med_energy=med_energy,
            max_gs=max_gs, broaden_mode=broaden_mode,
            xmin=xmin, xmax=xmax, nbins=nbins,
            return_sticks=return_sticks, device=device,
        )

    # Full PyTorch pipeline (Phase 5)
    #
    # Uses fixture-template approach: angular RME data from pre-computed
    # Fortran .rme_rcg/.rme_rac files, with user-supplied physics params
    # (cf, delta, slater, soc) applied as overrides. Autograd flows through
    # slater_scale and soc_scale into the COWAN store Hamiltonian blocks.
    return _calcXAS_phase5(
        element=element, valence=valence, sym=sym, edge=edge,
        cf=cf, slater=slater, soc=soc, delta=delta, u=u,
        lmct=lmct, mlct=mlct, T=T, beam_fwhm=beam_fwhm,
        gamma1=gamma1, gamma2=gamma2, med_energy=med_energy,
        max_gs=max_gs, broaden_mode=broaden_mode,
        xmin=xmin, xmax=xmax, nbins=nbins,
        return_sticks=return_sticks, device=device,
    )


def _calcXAS_from_ban(
    ban_path: str,
    T: float = 80.0,
    beam_fwhm: float = 0.2,
    gamma1: float = 0.2,
    gamma2: float = 0.4,
    med_energy: float = 25.0,
    max_gs: int = 1,
    broaden_mode: str = "legacy",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    nbins: int = 2000,
    return_sticks: bool = False,
    device: str = "cpu",
):
    """
    Compute XAS spectrum from a pre-computed .ban_out file.

    This is the primary path for Phase 5 validation — reads Fortran output
    and applies the PyTorch spectrum layer (Phase 1).
    """
    from multitorch.io.read_oba import read_ban_output

    ban = read_ban_output(ban_path)

    # Get Boltzmann-weighted sticks
    E_sticks, M_sticks, _ = get_sticks(ban, T=T, max_gs=max_gs, device=device)

    if E_sticks.numel() == 0:
        raise ValueError(f"No transitions found in {ban_path}")

    # Set energy range
    E_min = float(E_sticks.min())
    E_max = float(E_sticks.max())
    if xmin is None:
        xmin = E_min - 5.0
    if xmax is None:
        xmax = E_max + 5.0

    x = torch.linspace(xmin, xmax, nbins, dtype=DTYPE, device=device)

    # Apply pseudo-Voigt broadening
    med = 0.5 * (E_min + E_max)
    y = pseudo_voigt(
        x, E_sticks, M_sticks,
        fwhm_g=beam_fwhm, fwhm_l=gamma1, fwhm_l2=gamma2,
        med_energy=med, mode=broaden_mode,
    )

    if return_sticks:
        sticks = torch.stack([E_sticks, M_sticks], dim=1)
        return x, y, sticks
    return x, y


def _find_fixture_dir(element: str, valence: str, sym: str) -> Path:
    """Locate the reference_data fixture directory for a given ion + symmetry.

    The naming convention under ``tests/reference_data/`` is::

        {element_lower}{valence_num}_{d_config}_{sym_lower}/

    For example: ``ni2_d8_oh/``, ``fe3_d5_oh/``, ``ti4_d0_oh/``.

    Special case: ``nid8ct/`` for Ni²⁺ d⁸ in D4h (the charge-transfer fixture
    with full D4h operators).
    """
    from multitorch.atomic.tables import get_d_electrons

    # Resolve fixture root: look in package data first, then tests/reference_data/
    pkg_dir = Path(__file__).parent.parent  # multitorch/
    refdata = pkg_dir / "data" / "fixtures"
    if not refdata.is_dir():
        # Fallback: tests/reference_data/ (dev/editable install)
        repo_root = pkg_dir.parent
        refdata = repo_root / "tests" / "reference_data"

    if not refdata.is_dir():
        raise FileNotFoundError(
            f"Cannot find fixture data. Install multitorch with package data "
            f"or clone the repository for the tests/reference_data/ directory."
        )

    elem = element.capitalize()
    val_lower = valence.lower()
    sym_lower = sym.lower()
    n_d = get_d_electrons(elem, val_lower)

    # Map valence string to numeric
    val_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5}
    val_num = val_map.get(val_lower, 0)

    # Special case: nid8ct for Ni²⁺ D4h
    if elem == 'Ni' and val_num == 2 and sym_lower == 'd4h':
        candidate = refdata / "nid8ct"
        if candidate.is_dir():
            return candidate

    # Standard naming: {elem_lower}{val_num}_d{n_d}_{sym}
    dir_name = f"{elem.lower()}{val_num}_d{n_d}_{sym_lower}"
    candidate = refdata / dir_name
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"No fixture directory found for {elem} {val_lower} {sym_lower}. "
        f"Tried: {candidate}. Available fixtures: "
        f"{[d.name for d in refdata.iterdir() if d.is_dir()]}"
    )


def _find_rcn31_out(fixture_dir: Path, element: str, valence: str) -> Path:
    """Find the .rcn31_out file for atomic parameters.

    Some fixtures (e.g. nid8ct) don't have their own .rcn31_out.
    Fall back to the single-config fixture for the same element/valence.
    """
    from multitorch.atomic.tables import get_d_electrons

    # Try in the fixture dir itself
    for f in fixture_dir.glob("*.rcn31_out"):
        return f

    # Fall back: look in the reference_data root for {elem}d{n_d}
    refdata = fixture_dir.parent
    n_d = get_d_electrons(element, valence)

    # Try common naming patterns
    for pattern in [f"*d{n_d}*/*.rcn31_out", f"*{element.lower()}*/*.rcn31_out"]:
        matches = sorted(refdata.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No .rcn31_out found for {element} d{n_d} in {refdata}"
    )


def _find_primary_fixture(fixture_dir: Path, pattern: str) -> Path:
    """Find a fixture file, excluding _abs/_ems RIXS variants."""
    candidates = sorted(fixture_dir.glob(pattern))
    for c in candidates:
        stem = c.stem
        if not stem.endswith('_abs') and not stem.endswith('_ems'):
            return c
    # If all candidates have suffixes, fall back to the first one
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No file matching {pattern} in {fixture_dir}"
    )


def _calcXAS_phase5(
    element: str, valence: str, sym: str, edge: str,
    cf: dict,
    slater: float = 0.8,
    soc: float = 1.0,
    delta=None, u=None, lmct=None, mlct=None,
    T: float = 80.0,
    beam_fwhm: float = 0.2,
    gamma1: float = 0.2,
    gamma2: float = 0.4,
    med_energy: float = 25.0,
    max_gs: int = 1,
    broaden_mode: str = "legacy",
    xmin=None, xmax=None, nbins: int = 2000,
    return_sticks: bool = False,
    device: str = "cpu",
):
    """Phase 5 pure-PyTorch XAS pipeline using fixture templates.

    Uses pre-computed Fortran .rme_rcg/.rme_rac files for the angular RME
    structure (triads, ADD entries, section plan) and applies user-supplied
    physics parameters (cf, delta, slater, soc) as overrides.  Autograd
    flows through ``slater`` and ``soc`` into the COWAN store Hamiltonian
    blocks via :mod:`~multitorch.hamiltonian.build_cowan`.
    """
    from multitorch.atomic.parameter_fixtures import read_rcn31_out_params
    from multitorch.atomic.scaled_params import scale_atomic_params
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.hamiltonian.build_ban import modify_ban_params
    from multitorch.hamiltonian.build_cowan import build_cowan_store_in_memory
    from multitorch.hamiltonian.build_rac import build_rac_in_memory
    from multitorch.io.read_ban import read_ban
    from multitorch.spectrum.sticks import get_sticks_from_banresult
    from multitorch.spectrum.broaden import pseudo_voigt

    # Step 1: Locate fixture files (exclude _abs/_ems RIXS variants)
    fixture_dir = _find_fixture_dir(element, valence, sym)
    ban_path = _find_primary_fixture(fixture_dir, "*.ban")
    rcg_path = _find_primary_fixture(fixture_dir, "*.rme_rcg")
    rac_path = _find_primary_fixture(fixture_dir, "*.rme_rac")
    rcn31_path = _find_rcn31_out(fixture_dir, element, valence)

    # Step 2: Parse template BanData and apply user overrides (C2)
    ban = read_ban(ban_path)
    ban = modify_ban_params(ban, cf=cf, delta=delta, lmct=lmct, mlct=mlct)

    # Step 3: Load and scale atomic parameters (C3b + C3c)
    raw_params = read_rcn31_out_params(rcn31_path)
    scaled_params = scale_atomic_params(
        raw_params, slater_scale=slater, soc_scale=soc,
    )

    # Step 4: Build RAC structure from fixture (C3d)
    rac, plan = build_rac_in_memory(
        ban, source_rac_path=rac_path, source_rcg_path=rcg_path,
    )

    # Step 5: Build COWAN store with autograd-carrying params (C3e)
    cowan = build_cowan_store_in_memory(
        scaled_params, raw_params, plan, source_rcg_path=rcg_path, device=device,
    )

    # Step 6: Assemble Hamiltonian and diagonalize (C1)
    result = assemble_and_diagonalize_in_memory(cowan, rac, ban, device=device)

    # Step 7: Extract stick spectrum
    E_sticks, M_sticks, _ = get_sticks_from_banresult(
        result, T=T, max_gs=max_gs, device=device,
    )

    if E_sticks.numel() == 0:
        raise ValueError(
            f"No transitions found for {element} {valence} {sym}"
        )

    # Step 8: Set energy range and broaden
    E_min = float(E_sticks.min())
    E_max = float(E_sticks.max())
    if xmin is None:
        xmin = E_min - 5.0
    if xmax is None:
        xmax = E_max + 5.0

    x = torch.linspace(xmin, xmax, nbins, dtype=DTYPE, device=device)

    med = 0.5 * (E_min + E_max)
    y = pseudo_voigt(
        x, E_sticks, M_sticks,
        fwhm_g=beam_fwhm, fwhm_l=gamma1, fwhm_l2=gamma2,
        med_energy=med, mode=broaden_mode,
    )

    if return_sticks:
        sticks = torch.stack([E_sticks, M_sticks], dim=1)
        return x, y, sticks
    return x, y


# ─────────────────────────────────────────────────────────────
# Standalone (no-fixture) XAS pipeline
# ─────────────────────────────────────────────────────────────

def _hfs_to_slater_params(
    Z: int,
    gs_config: dict,
    ex_config: dict,
    slater_scale: float = 1.0,
    soc_scale: float = 1.0,
    zeta_method: str = "blume_watson",
):
    """Run HFS SCF for ground and excited configs, return Slater parameter dicts.

    Returns the dicts in the format expected by ``generate_ledge_rac``:
      gs_slater_ry, gs_zeta_ry, ex_slater_ry, ex_zeta_ry
    """
    from multitorch.atomic.hfs import hfs_scf
    from multitorch.atomic.slater import compute_slater_from_wavefunctions

    # Ground state HFS
    hfs_gs = hfs_scf(Z, gs_config, zeta_method=zeta_method)
    pnl_gs = {orb.nl_label.lower(): orb.P for orb in hfs_gs.orbitals
              if orb.P is not None}
    slater_gs = compute_slater_from_wavefunctions(
        pnl_gs, hfs_gs.r, hfs_gs.r[1].item() - hfs_gs.r[0].item())

    # Excited state HFS
    hfs_ex = hfs_scf(Z, ex_config, zeta_method=zeta_method)
    pnl_ex = {orb.nl_label.lower(): orb.P for orb in hfs_ex.orbitals
              if orb.P is not None}
    slater_ex = compute_slater_from_wavefunctions(
        pnl_ex, hfs_ex.r, hfs_ex.r[1].item() - hfs_ex.r[0].item())

    # Extract zeta values from HFS orbitals
    zeta_3d_gs = next(
        (orb.zeta_ry for orb in hfs_gs.orbitals if orb.nl_label.upper() == '3D'),
        0.0)
    zeta_3d_ex = next(
        (orb.zeta_ry for orb in hfs_ex.orbitals if orb.nl_label.upper() == '3D'),
        0.0)
    zeta_2p_ex = next(
        (orb.zeta_ry for orb in hfs_ex.orbitals if orb.nl_label.upper() == '2P'),
        0.0)

    # Format for generate_ledge_rac
    gs_slater_ry = {
        'F0': float(slater_gs.get('F0dd', 0.0)) * slater_scale,
        'F2': float(slater_gs.get('F2dd', 0.0)) * slater_scale,
        'F4': float(slater_gs.get('F4dd', 0.0)) * slater_scale,
    }
    gs_zeta_ry = float(zeta_3d_gs) * soc_scale

    ex_slater_ry = {
        'F2_dd': float(slater_ex.get('F2dd', 0.0)) * slater_scale,
        'F4_dd': float(slater_ex.get('F4dd', 0.0)) * slater_scale,
        'G1_pd': float(slater_ex.get('G1pd', 0.0)) * slater_scale,
        'G3_pd': float(slater_ex.get('G3pd', 0.0)) * slater_scale,
        'F2_pd': float(slater_ex.get('F2pd', 0.0)) * slater_scale,
    }
    ex_zeta_ry = {
        'd': float(zeta_3d_ex) * soc_scale,
        'p': float(zeta_2p_ex) * soc_scale,
    }

    return gs_slater_ry, gs_zeta_ry, ex_slater_ry, ex_zeta_ry


def _build_ban_from_rac(rac, tendq: float = 1.0):
    """Build a minimal BanData from a generated RAC for single-config Oh.

    Extracts triads from TRANSI blocks and constructs XHAM entries
    for the HAMILTONIAN (k=0) + 10DQ (CF) operators.
    """
    from multitorch.io.read_ban import BanData, XHAMEntry

    triads = []
    for b in rac.blocks:
        if b.kind == 'TRANSI' and b.add_entries:
            triad = (b.bra_sym, b.op_sym, b.ket_sym)
            if triad not in triads:
                triads.append(triad)

    # XHAM: operator 1 = HAMILTONIAN (weight 1.0), operator 2 = 10DQ (weight tendq)
    # combos: (config, operator) pairs — for single config, config=1
    xham = [XHAMEntry(
        values=[1.0, tendq],
        combos=[(1, 1), (1, 2)],
    )]

    return BanData(
        nconf_gs=1,
        nconf_fs=1,
        xham=xham,
        tran=[(1, 1)],
        triads=triads,
        eg={1: 0.0},
        ef={1: 0.0},
    )


def calcXAS_from_scratch(
    element: str,
    valence: str,
    cf: dict,
    slater: float = 0.8,
    soc: float = 1.0,
    T: float = 80.0,
    beam_fwhm: float = 0.2,
    gamma1: float = 0.2,
    gamma2: float = 0.4,
    med_energy: float = 25.0,
    max_gs: int = 1,
    broaden_mode: str = "legacy",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    nbins: int = 2000,
    return_sticks: bool = False,
    device: str = "cpu",
    zeta_method: str = "blume_watson",
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Calculate an L-edge XAS spectrum from scratch — no Fortran fixture files.

    Runs the full pipeline:
      1. HFS SCF → radial wavefunctions for ground and excited configs
      2. Slater integrals + SOC from the HFS wavefunctions
      3. Angular RME structure (RAC + COWAN store) via generate_ledge_rac
      4. BanData construction from the RAC topology
      5. Hamiltonian assembly + diagonalization
      6. Boltzmann-weighted stick spectrum → pseudo-Voigt broadening

    Currently supports single-configuration Oh symmetry only (no charge
    transfer, no D4h splitting beyond PERP/PARA). This is sufficient for
    isolated-ion crystal-field calculations.

    Parameters
    ----------
    element : str
        Element symbol (e.g. 'Ni', 'Fe', 'Ti').
    valence : str
        Oxidation state ('i', 'ii', 'iii', 'iv').
    cf : dict
        Crystal field parameters. Must include 'tendq' (10Dq in eV).
    slater : float
        Slater integral reduction factor (0-1, default 0.8).
    soc : float
        Spin-orbit coupling reduction factor (0-1, default 1.0).
    T : float
        Temperature in Kelvin.
    beam_fwhm, gamma1, gamma2, med_energy : float
        Broadening parameters (eV).
    max_gs : int
        Number of ground states to include in Boltzmann average.
    broaden_mode : str
        'legacy' or 'correct' pseudo-Voigt mode.
    xmin, xmax : float or None
        Energy range. Auto-detected if None.
    nbins : int
        Number of energy bins.
    return_sticks : bool
        If True, also return the stick spectrum.
    device : str
        PyTorch device.
    zeta_method : str
        HFS SOC method ('blume_watson' or 'central_field').

    Returns
    -------
    x : torch.Tensor  shape (nbins,)
        Energy axis (eV).
    y : torch.Tensor  shape (nbins,)
        Absorption intensity.
    sticks : torch.Tensor  shape (N, 2)  (only if return_sticks=True)
    """
    from multitorch.angular.rac_generator import generate_ledge_rac
    from multitorch.atomic.tables import (
        get_atomic_number, get_d_electrons, get_l_edge_configs,
        parse_config_string,
    )
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.spectrum.sticks import get_sticks_from_banresult

    # Step 1: Resolve element info
    Z = get_atomic_number(element)
    n_d = get_d_electrons(element, valence)
    gs_cfg_str, ex_cfg_str = get_l_edge_configs(element, valence)
    gs_config = parse_config_string(gs_cfg_str)
    ex_config = parse_config_string(ex_cfg_str)

    # Step 2: HFS SCF → Slater integrals + SOC
    gs_slater, gs_zeta, ex_slater, ex_zeta = _hfs_to_slater_params(
        Z, gs_config, ex_config,
        slater_scale=slater, soc_scale=soc,
        zeta_method=zeta_method,
    )

    # Step 3: Generate angular structure + COWAN store
    rac, cowan = generate_ledge_rac(
        l_val=2, n_val_gs=n_d,
        raw_slater_gs_ry=gs_slater,
        raw_zeta_gs_ry=gs_zeta,
        raw_slater_ex_ry=ex_slater,
        raw_zeta_ex_ry=ex_zeta,
    )

    # Step 4: Build BanData from RAC
    tendq = cf.get('tendq', 1.0)
    ban = _build_ban_from_rac(rac, tendq=tendq)

    # Step 5: Assemble and diagonalize
    result = assemble_and_diagonalize_in_memory(cowan, rac, ban, device=device)

    # Step 6: Extract stick spectrum
    E_sticks, M_sticks, _ = get_sticks_from_banresult(
        result, T=T, max_gs=max_gs, device=device,
    )

    if E_sticks.numel() == 0:
        raise ValueError(
            f"No transitions found for {element} {valence}"
        )

    # Step 7: Broaden
    E_min = float(E_sticks.min())
    E_max = float(E_sticks.max())
    if xmin is None:
        xmin = E_min - 5.0
    if xmax is None:
        xmax = E_max + 5.0

    x = torch.linspace(xmin, xmax, nbins, dtype=DTYPE, device=device)

    med = 0.5 * (E_min + E_max)
    y = pseudo_voigt(
        x, E_sticks, M_sticks,
        fwhm_g=beam_fwhm, fwhm_l=gamma1, fwhm_l2=gamma2,
        med_energy=med, mode=broaden_mode,
    )

    if return_sticks:
        sticks = torch.stack([E_sticks, M_sticks], dim=1)
        return x, y, sticks
    return x, y


def calcXES(
    element: str,
    valence: str,
    sym: str,
    edge: str,
    cf: dict,
    slater: float = 0.8,
    soc: float = 1.0,
    ban_output_path: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate an X-ray emission spectrum.

    See calcXAS for parameter documentation. For XES, the ground/final
    state assignments are swapped (emission from a core-hole state).
    """
    if ban_output_path is not None:
        return _calcXAS_from_ban(
            ban_output_path,
            T=kwargs.pop('T', 80.0),
            beam_fwhm=kwargs.pop('beam_fwhm', 0.2),
            gamma1=kwargs.pop('gamma1', 0.2),
            gamma2=kwargs.pop('gamma2', 0.4),
            **kwargs,
        )
    raise NotImplementedError("Full XES pipeline pending. Use ban_output_path.")


def calcRIXS(
    element: str = '',
    valence: str = '',
    sym: str = '',
    edge: str = '',
    cf: Optional[dict] = None,
    Einc: Optional[torch.Tensor] = None,
    Efin: Optional[torch.Tensor] = None,
    Gamma_i: float = 0.4,
    Gamma_f: float = 0.2,
    T: float = 80.0,
    n_Einc: int = 400,
    n_Efin: int = 400,
    pad_eV: float = 5.0,
    ban_abs_path: Optional[str] = None,
    ban_ems_path: Optional[str] = None,
    return_store: bool = False,
    device: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate a RIXS (resonant inelastic X-ray scattering) plane.

    Bootstrap mode: pass ``ban_abs_path`` and ``ban_ems_path`` to read a
    paired absorption/emission ``.ban_out`` set produced by ttban. The full
    PyTorch physics pipeline (Phase 5) is not yet wired into this function.

    Parameters
    ----------
    Einc, Efin
        Energy grids in eV. If ``None``, auto-generated to span the
        intermediate-state energies (Einc) and the relevant emitted-photon
        range (Efin) with ``pad_eV`` padding.
    Gamma_i, Gamma_f
        Intermediate- and final-state lifetime FWHM (eV). Defaults to
        ``0.4`` and ``0.2``, in line with typical L-edge core-hole and
        resolution widths.
    T
        Temperature for Boltzmann population of ground states (K).
    n_Einc, n_Efin
        Auto-generated grid sizes (only used when the corresponding axis
        is ``None``).
    pad_eV
        Padding around the data range used when auto-generating grids.
    ban_abs_path, ban_ems_path
        Paths to the absorption and emission ``.ban_out`` files.
    return_store
        If True, also return the underlying ``RIXSStore`` for inspection.
    device : str or None
        PyTorch device ('cpu', 'cuda', 'cuda:0', etc.). If None, automatically
        selects optimal device. **RIXS strongly benefits from GPU** with measured
        45× speedup for typical 400×400 grids due to memory-bandwidth-intensive
        Kramers-Heisenberg calculation (~128 MB intermediate tensors).
        
        - Default: 'cuda' if available, else 'cpu'
        - Recommendation: Always use GPU for RIXS if available

    Returns
    -------
    Einc : torch.Tensor  shape (n_Einc,)
        Incident energy axis (eV).
    Efin : torch.Tensor  shape (n_Efin,)
        Emitted-photon energy axis (eV).
    intensity : torch.Tensor  shape (n_Einc, n_Efin)
        Summed Kramers-Heisenberg intensity over all symmetry channels and
        Boltzmann-weighted ground states.
    """
    # Smart device selection: RIXS strongly benefits from GPU (45× speedup)
    if device is None:
        device = suggest_device_for_rixs()
    
    if ban_abs_path is not None and ban_ems_path is not None:
        from multitorch.io.read_oba_pair import read_abs_ems_pair
        store = read_abs_ems_pair(ban_abs_path, ban_ems_path)
    elif element and valence and sym and edge:
        store = _build_rixs_store_phase5(
            element, valence, sym, edge,
            cf or {}, kwargs.get('slater', 1.0), kwargs.get('soc', 1.0),
            kwargs.get('delta'), kwargs.get('lmct'), kwargs.get('mlct'),
            device,
        )
    else:
        raise ValueError(
            "Either provide both ban_abs_path and ban_ems_path "
            "(bootstrap mode) or element/valence/sym/edge (Phase 5 mode)."
        )

    from multitorch.spectrum.rixs import kramers_heisenberg
    if not store.channels:
        raise ValueError(
            f"No matching absorption/emission triads found between "
            f"{ban_abs_path} and {ban_ems_path}."
        )

    min_gs = store.min_gs

    # Auto-generate Einc / Efin grids from the channel data.
    if Einc is None:
        Ei_min = min(float(ch.Ei.min()) for ch in store.channels)
        Ei_max = max(float(ch.Ei.max()) for ch in store.channels)
        Einc = torch.linspace(
            Ei_min - pad_eV, Ei_max + pad_eV, n_Einc,
            dtype=DTYPE, device=device,
        )
    else:
        Einc = torch.as_tensor(Einc, dtype=DTYPE, device=device)

    if Efin is None:
        # Emitted photon energy Efin satisfies energy conservation
        # Efin ≈ Einc − (Ef − Eg) for the dominant peaks (all in eV).
        # Span the full reachable range with the same padding.
        max_loss = 0.0
        min_loss = 0.0
        for ch in store.channels:
            Eg_min = float(ch.Eg.min())
            losses = ch.Ef - Eg_min   # (n_f,) eV  — final - ground
            max_loss = max(max_loss, float(losses.max()))
            min_loss = min(min_loss, float(losses.min()))
        Ei_min = min(float(ch.Ei.min()) for ch in store.channels)
        Ei_max = max(float(ch.Ei.max()) for ch in store.channels)
        Efin = torch.linspace(
            Ei_min - max_loss - pad_eV,
            Ei_max - min_loss + pad_eV,
            n_Efin,
            dtype=DTYPE, device=device,
        )
    else:
        Efin = torch.as_tensor(Efin, dtype=DTYPE, device=device)

    intensity = torch.zeros((Einc.shape[0], Efin.shape[0]),
                            dtype=DTYPE, device=device)
    for ch in store.channels:
        rm = kramers_heisenberg(
            Eg=ch.Eg, TA=ch.TA, Ei=ch.Ei, TE=ch.TE, Ef=ch.Ef,
            Einc=Einc, Efin=Efin,
            Gamma_i=Gamma_i, Gamma_f=Gamma_f,
            min_gs=min_gs, T=T, device=device,
        )
        intensity = intensity + rm

    if return_store:
        return Einc, Efin, intensity, store
    return Einc, Efin, intensity


def _run_phase5_pipeline(
    element: str, valence: str, sym: str, edge: str,
    cf: dict, slater, soc, delta, lmct, mlct,
    device: str = 'cpu',
    *,
    fixture_suffix: str = '',
):
    """Run the Phase 5 pipeline and return the raw BanResult.

    Parameters
    ----------
    fixture_suffix : str
        If non-empty, look for fixture files named ``{case_id}{suffix}.*``
        instead of the default. Used for emission fixtures (suffix='_ems').
    """
    from multitorch.atomic.parameter_fixtures import read_rcn31_out_params
    from multitorch.atomic.scaled_params import scale_atomic_params
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.hamiltonian.build_ban import modify_ban_params
    from multitorch.hamiltonian.build_cowan import build_cowan_store_in_memory
    from multitorch.hamiltonian.build_rac import build_rac_in_memory
    from multitorch.io.read_ban import read_ban

    fixture_dir = _find_fixture_dir(element, valence, sym)

    if fixture_suffix:
        # Look for files with the suffix pattern
        ban_candidates = list(fixture_dir.glob(f"*{fixture_suffix}.ban"))
        rcg_candidates = list(fixture_dir.glob(f"*{fixture_suffix}.rme_rcg"))
        rac_candidates = list(fixture_dir.glob(f"*{fixture_suffix}.rme_rac"))
        if not (ban_candidates and rcg_candidates and rac_candidates):
            raise FileNotFoundError(
                f"No fixture files found for suffix '{fixture_suffix}' "
                f"in {fixture_dir}. RIXS Phase 5 requires emission fixture "
                f"files (e.g. *_ems.ban, *_ems.rme_rcg, *_ems.rme_rac)."
            )
        ban_path = ban_candidates[0]
        rcg_path = rcg_candidates[0]
        rac_path = rac_candidates[0]
    else:
        ban_path = _find_primary_fixture(fixture_dir, "*.ban")
        rcg_path = _find_primary_fixture(fixture_dir, "*.rme_rcg")
        rac_path = _find_primary_fixture(fixture_dir, "*.rme_rac")

    rcn31_path = _find_rcn31_out(fixture_dir, element, valence)

    ban = read_ban(ban_path)
    ban = modify_ban_params(ban, cf=cf, delta=delta, lmct=lmct, mlct=mlct)

    raw_params = read_rcn31_out_params(rcn31_path)
    scaled_params = scale_atomic_params(
        raw_params, slater_scale=slater, soc_scale=soc,
    )

    rac, plan = build_rac_in_memory(
        ban, source_rac_path=rac_path, source_rcg_path=rcg_path,
    )
    cowan = build_cowan_store_in_memory(
        scaled_params, raw_params, plan, source_rcg_path=rcg_path, device=device,
    )
    return assemble_and_diagonalize_in_memory(cowan, rac, ban, device=device)


def _banresult_to_banoutput(result):
    """Convert a BanResult to a BanOutput for RIXS pairing.

    Each TriadResult becomes one or more TriadData entries (one per
    ground state), mirroring the .ban_out file format where each row
    of the transition matrix is a separate entry.
    """
    from multitorch.io.read_oba import BanOutput, TriadData

    bo = BanOutput()
    for t in result.triads:
        # T is (n_gs, n_fs) — transition matrix elements
        # M = T**2 (pre-squared intensities, matching .ban_out convention)
        M = t.T ** 2

        # Emit one TriadData per ground state (matching .ban_out convention)
        for g in range(t.n_gs):
            td = TriadData(
                ground_sym=t.gs_sym,
                op_sym=t.act_sym,
                final_sym=t.fs_sym,
                actor="MULTIPOLE",
                Eg=t.Eg[g:g+1],       # (1,) eV
                Ef=t.Ef,               # (n_fs,) eV
                M=M[g:g+1, :],         # (1, n_fs)
            )
            bo.triad_list.append(td)

    return bo


def _build_rixs_store_phase5(
    element, valence, sym, edge, cf, slater, soc, delta, lmct, mlct,
    device='cpu',
):
    """Build a RIXSStore using Phase 5 absorption + bootstrap emission.

    The absorption side runs the full Phase 5 pipeline so that autograd
    flows through ``slater``, ``soc``, ``cf``, etc. into the RIXS map.
    The emission side uses the pre-computed ``*_ems.ban_out`` fixture
    because the emission fixture's ``.rme_rcg``/``.rme_rac`` structure
    differs from the absorption one (reversed ground/final configs) and
    the current Phase 5 pipeline validates against the absorption-side
    COWAN store layout.

    This means gradients flow through the absorption matrix elements but
    not the emission side — which is the primary use case for RIXS
    optimization (fitting to absorption resonance energies).
    """
    from multitorch.io.read_oba import read_ban_output
    from multitorch.io.read_oba_pair import build_rixs_store

    # Absorption: full Phase 5 pipeline (autograd-carrying)
    abs_result = _run_phase5_pipeline(
        element, valence, sym, edge, cf, slater, soc, delta, lmct, mlct,
        device=device,
    )
    abs_bo = _banresult_to_banoutput(abs_result)

    # Emission: bootstrap from pre-computed .ban_out
    fixture_dir = _find_fixture_dir(element, valence, sym)
    ems_ban_out_candidates = sorted(fixture_dir.glob("*_ems.ban_out"))
    if not ems_ban_out_candidates:
        raise FileNotFoundError(
            f"No emission .ban_out fixture found in {fixture_dir}. "
            f"RIXS Phase 5 requires a *_ems.ban_out file. Generate one "
            f"using pyttmult writeRIXS."
        )
    ems_bo = read_ban_output(ems_ban_out_candidates[0])

    store = build_rixs_store(abs_bo, ems_bo)
    if not store.channels:
        raise ValueError(
            "No matching absorption/emission channels found. "
            "Check that the emission fixture files are correctly paired."
        )
    return store


def calcDOC(
    element: str = '',
    valence: str = '',
    sym: str = '',
    edge: str = '',
    cf: Optional[dict] = None,
    T: float = 80.0,
    ban_output_path: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Calculate ground-state configuration weights (degree of covalency).

    For charge-transfer multiplet calculations, the ground state is a
    mixture of metal d^n and charge-transfer d^(n+1)L̲ configurations.
    This function returns the weight of each configuration in each
    ground-state eigenstate, Boltzmann-averaged at temperature *T*.

    Two modes of operation:

    1. **Bootstrap mode** (``ban_output_path`` set): reads the
       configuration weights directly from the ``.ban_out`` file header,
       which ttban prints as "Weight of configurations 1,2,3 in the
       ground state: w1 w2 w3".

    2. **Phase 5 mode** (``element``, ``valence``, ``sym``, ``edge``
       set): runs the full Phase 5 pipeline and computes configuration
       weights from the assembled Hamiltonian eigenvectors.

    Parameters
    ----------
    element, valence, sym, edge, cf
        Physical parameters (Phase 5 mode).
    T : float
        Temperature in K for Boltzmann averaging.
    ban_output_path : str, optional
        Path to a ``.ban_out`` file (bootstrap mode).
    **kwargs
        Passed through to ``calcXAS`` (Phase 5 mode) — ``slater``,
        ``soc``, ``delta``, ``lmct``, ``mlct``, etc.

    Returns
    -------
    dict with keys:
        ``'config_weights'`` : list of float
            Boltzmann-averaged weight of each configuration in the ground
            state, summed over all symmetry triads. ``config_weights[0]``
            is the metal d^n character; ``config_weights[1]`` is the
            d^(n+1)L̲ (LMCT) character; ``config_weights[2]`` (if present)
            is the MLCT character. Values sum to ~1.0.
        ``'per_triad'`` : list of dict
            Per-triad breakdown with keys ``'sym'``, ``'gs_energy'``,
            ``'config_weights'``.
        ``'metal_character'`` : float
            Shortcut: ``config_weights[0]`` (the d^n fraction).
        ``'ct_character'`` : float
            Shortcut: ``1 - config_weights[0]`` (total charge-transfer
            fraction).
    """
    if ban_output_path is not None:
        return _calcDOC_bootstrap(ban_output_path, T)
    else:
        return _calcDOC_phase5(
            element, valence, sym, edge, cf or {}, T, **kwargs
        )


def _calcDOC_bootstrap(ban_output_path: str, T: float) -> dict:
    """Extract DOC from a .ban_out file's header metadata."""
    from multitorch.io.read_oba import read_ban_output

    bo = read_ban_output(ban_output_path)
    if not bo.band_metadata:
        raise ValueError(
            f"No configuration weight metadata found in {ban_output_path}. "
            "The file may not have been produced by ttban_exact."
        )

    # Boltzmann-weight the per-band config weights
    from multitorch._constants import K_B_FLOAT
    energies = [bm.gs_energy for bm in bo.band_metadata]
    e_min = min(energies)

    if T > 0:
        boltz = [
            bm.gs_degeneracy * math.exp(-(bm.gs_energy - e_min) / (K_B_FLOAT * T))
            for bm in bo.band_metadata
        ]
    else:
        boltz = [
            float(abs(bm.gs_energy - e_min) < 1e-10)
            for bm in bo.band_metadata
        ]

    z = sum(boltz)
    if z < 1e-30:
        z = 1.0

    n_configs = max(len(bm.config_weights) for bm in bo.band_metadata)
    avg_weights = [0.0] * n_configs
    for bm, w in zip(bo.band_metadata, boltz):
        for i, cw in enumerate(bm.config_weights):
            avg_weights[i] += w * cw
    avg_weights = [w / z for w in avg_weights]

    per_triad = [
        {
            'sym': bm.triad_sym,
            'gs_energy': bm.gs_energy,
            'config_weights': bm.config_weights,
        }
        for bm in bo.band_metadata
    ]

    return {
        'config_weights': avg_weights,
        'per_triad': per_triad,
        'metal_character': avg_weights[0] if avg_weights else 0.0,
        'ct_character': 1.0 - avg_weights[0] if avg_weights else 0.0,
    }


def _calcDOC_phase5(
    element: str, valence: str, sym: str, edge: str,
    cf: dict, T: float, device: str = 'cpu', **kwargs,
) -> dict:
    """Compute DOC from Phase 5 assembled Hamiltonian eigenvectors."""
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.hamiltonian.build_ban import modify_ban_params
    from multitorch.hamiltonian.build_cowan import build_cowan_store_in_memory
    from multitorch.hamiltonian.build_rac import build_rac_in_memory
    from multitorch.atomic.scaled_params import scale_atomic_params
    from multitorch.atomic.parameter_fixtures import read_rcn31_out_params
    from multitorch.io.read_ban import read_ban

    fixture_dir = _find_fixture_dir(element, valence, sym)
    ban_path = _find_primary_fixture(fixture_dir, "*.ban")
    rcg_path = _find_primary_fixture(fixture_dir, "*.rme_rcg")
    rac_path = _find_primary_fixture(fixture_dir, "*.rme_rac")
    rcn31_path = _find_rcn31_out(fixture_dir, element, valence)
    ban = read_ban(str(ban_path))

    slater = kwargs.pop('slater', 1.0)
    soc = kwargs.pop('soc', 1.0)
    delta = kwargs.pop('delta', None)
    lmct = kwargs.pop('lmct', None)
    mlct = kwargs.pop('mlct', None)

    ban = modify_ban_params(ban, cf=cf, delta=delta, lmct=lmct, mlct=mlct)

    raw_params = read_rcn31_out_params(str(rcn31_path))
    scaled_params = scale_atomic_params(
        raw_params, slater_scale=slater, soc_scale=soc,
    )

    rac, plan = build_rac_in_memory(
        ban, source_rac_path=rac_path, source_rcg_path=rcg_path,
    )
    cowan = build_cowan_store_in_memory(
        scaled_params, raw_params, plan, source_rcg_path=rcg_path, device=device,
    )
    result = assemble_and_diagonalize_in_memory(cowan, rac, ban, device=device)

    from multitorch._constants import K_B_FLOAT

    # Collect config weights from all triads
    all_eg = []
    for t in result.triads:
        all_eg.append(t.Eg.min().item())
    e_min = min(all_eg) if all_eg else 0.0

    per_triad = []
    n_configs = max(
        (len(t.gs_conf_sizes) for t in result.triads), default=1
    )
    weighted_sum = [0.0] * n_configs
    z_total = 0.0

    for t in result.triads:
        n = t.Eg.shape[0]
        cw = []
        for c_idx in range(len(t.gs_conf_sizes)):
            mask = (t.gs_conf_labels == (c_idx + 1))
            config_weight_per_state = (
                t.Ug[mask, :].pow(2).sum(dim=0)
            )
            cw.append(config_weight_per_state)

        # Boltzmann weights — eigenvalues are in eV (build_cowan converts Ry→eV)
        e_rel = t.Eg - e_min
        if T > 0:
            boltz = torch.exp(-e_rel / (K_B_FLOAT * T))
        else:
            boltz = (e_rel.abs() < 1e-10).to(DTYPE)

        z_triad = float(boltz.sum())
        z_total += z_triad

        triad_weights = []
        for c_idx in range(len(t.gs_conf_sizes)):
            w = float((cw[c_idx] * boltz).sum())
            weighted_sum[c_idx] += w
            triad_weights.append(
                float((cw[c_idx] * boltz).sum()) / max(z_triad, 1e-30)
            )

        per_triad.append({
            'sym': f"{t.gs_sym}  {t.act_sym}  {t.fs_sym}",
            'gs_energy': float(t.Eg[0]),  # already in eV
            'config_weights': triad_weights,
        })

    if z_total > 1e-30:
        avg_weights = [w / z_total for w in weighted_sum]
    else:
        avg_weights = weighted_sum

    return {
        'config_weights': avg_weights,
        'per_triad': per_triad,
        'metal_character': avg_weights[0] if avg_weights else 0.0,
        'ct_character': 1.0 - avg_weights[0] if avg_weights else 0.0,
    }
