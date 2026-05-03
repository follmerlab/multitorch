"""
Device selection utilities for optimal CPU/GPU dispatch.

Provides smart device selection based on operation characteristics and
problem size to maximize performance in parameter refinement workflows.
"""
import torch
from typing import Optional


def get_optimal_device(
    operation: str = "xas",
    matrix_dim: Optional[int] = None,
    n_sticks: Optional[int] = None,
    force_device: Optional[str] = None,
) -> str:
    """
    Select optimal device (CPU/GPU) based on operation characteristics.

    GPU acceleration benefits memory-bandwidth-intensive operations like
    RIXS 2D maps (45× speedup) and large broadening (41× speedup), but
    adds overhead for small matrix operations typical in 3d transition
    metal L-edge calculations.

    Parameters
    ----------
    operation : str
        Operation type: 'xas', 'rixs', 'broaden', 'eigh'
    matrix_dim : int or None
        Hamiltonian matrix dimension (for eigh operations)
    n_sticks : int or None
        Number of stick transitions (for broadening operations)
    force_device : str or None
        If provided, overrides automatic selection ('cpu', 'cuda', 'cuda:0', etc.)

    Returns
    -------
    device : str
        Device string suitable for torch.device()

    Notes
    -----
    Decision logic based on measured performance:

    - **RIXS**: Always GPU (45× speedup, ~128 MB memory for 400×400 grid)
    - **Broadening**: GPU if n_sticks > 1000 (41× speedup)
    - **Diagonalization**: GPU if matrix_dim >= 500 (4-10× speedup)
    - **Single L-edge XAS**: CPU for typical 3d TM (dim < 200, kernel overhead dominates)

    Examples
    --------
    >>> # RIXS calculation - automatically uses GPU
    >>> device = get_optimal_device('rixs')
    >>> # Returns 'cuda' if available, else 'cpu'

    >>> # Small L-edge spectrum - stays on CPU
    >>> device = get_optimal_device('xas', matrix_dim=17)
    >>> # Returns 'cpu' (typical Ni d8 Oh case)

    >>> # Large rare earth system - uses GPU
    >>> device = get_optimal_device('eigh', matrix_dim=1000)
    >>> # Returns 'cuda' if available
    """
    # If device explicitly specified, use it
    if force_device is not None:
        return force_device

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()

    # RIXS: Always prefer GPU (45× speedup measured)
    if operation == "rixs":
        return "cuda" if cuda_available else "cpu"

    # Broadening: GPU for large stick counts (41× speedup)
    if operation == "broaden":
        if n_sticks is not None and n_sticks > 1000:
            return "cuda" if cuda_available else "cpu"
        return "cpu"  # Small broadening: CPU optimal

    # Eigenvalue decomposition: GPU for large matrices
    if operation == "eigh":
        if matrix_dim is not None and matrix_dim >= 500:
            return "cuda" if cuda_available else "cpu"
        return "cpu"  # Small eigh: kernel launch overhead > compute

    # Single L-edge XAS: CPU optimal for typical 3d transition metals
    # (dim < 200, kernel overhead dominates)
    if operation == "xas":
        # Check if this is a large rare earth system
        if matrix_dim is not None and matrix_dim >= 500:
            return "cuda" if cuda_available else "cpu"
        return "cpu"  # Standard 3d TM: CPU optimal

    # Default: CPU for unknown operations
    return "cpu"


def suggest_device_for_xas(
    element: Optional[str] = None,
    valence: Optional[str] = None,
    force_device: Optional[str] = None,
) -> str:
    """
    Suggest optimal device for XAS calculation.

    Empirical guidance from a 2026-05-03 RTX 4090 vs Threadripper-MKL
    bench (`bench/AGENT_HANDOFF_GPU_EIGH.md` results, full-pipeline
    `calcXAS_cached`):

    | Configuration             | Max H block | GPU vs CPU |
    |---------------------------|-------------|------------|
    | Ni d8 oh                  | 4           | slower (overhead) |
    | Co d7 oh                  | small       | slower       |
    | Fe d6 oh                  | 81          | ~tied       |
    | Cr d3, Mn d5, V d2, Fe d5 | 100–200     | 3-4× faster |
    | Rare earths (4f)          | > 500       | 5-20× faster |

    Threshold heuristic: max H block ≥ ~100 → GPU pays off.
    For 3d transition metals, this maps to d4/d5/d6 (high-spin
    configurations with many LS terms).

    Parameters
    ----------
    element : str or None
        Element symbol (e.g. 'Ni', 'Fe', 'Ce')
    valence : str or None
        Oxidation state ('ii', 'iii', etc.)
    force_device : str or None
        If provided, overrides automatic selection. Use 'auto' to
        re-trigger heuristic (same as None).

    Returns
    -------
    device : str
        'cuda' if GPU expected to help and available; 'cpu' otherwise.
    """
    if force_device is not None and force_device != "auto":
        return force_device

    # Need CUDA to even consider GPU
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
    except ImportError:
        return "cpu"

    # Element/valence lookup for empirically GPU-favorable cases.
    # All 3d high-spin d4/d5/d6 (large LS-term count → large blocks).
    GPU_FAVORS_3D = {
        ("V",  "iii"),  # d2 — moderate
        ("Cr", "iii"),  # d3
        ("Mn", "ii"),   # d5
        ("Mn", "iii"),  # d4
        ("Fe", "ii"),   # d6 — borderline
        ("Fe", "iii"),  # d5
    }
    # 4f rare earths: GPU always wins
    RARE_EARTHS = {"Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                   "Tb", "Dy", "Ho", "Er", "Tm", "Yb"}

    if element in RARE_EARTHS:
        return "cuda"
    if element is not None and valence is not None:
        if (element, valence.lower()) in GPU_FAVORS_3D:
            return "cuda"

    # Default: small/unknown systems use CPU (avoids overhead)
    return "cpu"


def suggest_device_for_rixs(force_device: Optional[str] = None) -> str:
    """
    Suggest optimal device for RIXS calculation.

    RIXS 2D maps are memory-bandwidth intensive with ~45× measured
    GPU speedup. Always recommend GPU if available.

    Parameters
    ----------
    force_device : str or None
        If provided, overrides automatic selection

    Returns
    -------
    device : str
        Recommended device string (typically 'cuda' if available)
    """
    if force_device is not None:
        return force_device

    return "cuda" if torch.cuda.is_available() else "cpu"
