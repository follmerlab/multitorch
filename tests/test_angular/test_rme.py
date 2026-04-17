"""
Tests for RME builder: unit tensor U^(k) and SPIN computation from CFP + Wigner 6j.

Validates against SHELL1 and SPIN1 blocks in tests/reference_data/nid8ct/nid8ct.rme_rcg.
"""
import pytest
import numpy as np
from pathlib import Path

REFDATA = Path(__file__).parent.parent / "reference_data" / "nid8ct"
BINPATH = None  # auto-detect ttmult/bin


def _get_terms_and_parents(l, n):
    """Build LSTerm lists from CFP tables."""
    from multitorch.angular.rme import LSTerm
    from multitorch.angular.cfp import get_cfp_block

    block_n = get_cfp_block(l, n, BINPATH)
    block_nm1 = get_cfp_block(l, n - 1, BINPATH)

    terms = [LSTerm(index=t.index, S=t.S, L=t.L, seniority=t.seniority,
                    label=f"{int(2*t.S+1)}{t.L_label}") for t in block_n.terms]
    parents = [LSTerm(index=t.index, S=t.S, L=t.L, seniority=t.seniority,
                      label=f"{int(2*t.S+1)}{t.L_label}") for t in block_nm1.terms]
    return terms, parents, block_n.cfp


def _load_ref_blocks(config_idx=0):
    """Load reference SHELL and SPIN blocks from .rme_rcg."""
    from multitorch.io.read_rme import read_rme_rcg

    ref = read_rme_rcg(REFDATA / "nid8ct.rme_rcg")
    shell_configs = [c for c in ref.configs
                     if any(b.operator.startswith('SHELL') for b in c.blocks)]
    cfg = shell_configs[config_idx]

    shell_ref = {}  # (k, J_bra, J_ket) -> np.ndarray
    spin_ref = {}   # (J_bra, J_ket) -> np.ndarray
    for b in cfg.blocks:
        k_val = int(b.op_sym.replace('+', '').replace('-', '').replace('^', ''))
        J_bra = float(int(b.bra_sym.replace('+', '').replace('-', '').replace('^', '')))
        J_ket = float(int(b.ket_sym.replace('+', '').replace('-', '').replace('^', '')))
        if b.operator.startswith('SHELL'):
            shell_ref[(k_val, J_bra, J_ket)] = b.matrix.numpy()
        elif b.operator.startswith('SPIN'):
            spin_ref[(J_bra, J_ket)] = b.matrix.numpy()
    return shell_ref, spin_ref


# ─── CFP term info tests ───────────────────────────────────

@pytest.mark.phase3
def test_d8_term_count():
    from multitorch.angular.cfp import get_cfp_block
    block = get_cfp_block(2, 8)
    assert block is not None
    assert block.nlt == 5
    assert len(block.terms) == 5


@pytest.mark.phase3
def test_d8_term_quantum_numbers():
    from multitorch.angular.cfp import get_cfp_block
    block = get_cfp_block(2, 8)
    terms = block.terms
    # d^8 = d^2 hole equivalent: 3F, 3P, 1G, 1D, 1S
    sl_pairs = [(t.S, t.L) for t in terms]
    assert (1.0, 3.0) in sl_pairs  # 3F
    assert (1.0, 1.0) in sl_pairs  # 3P
    assert (0.0, 4.0) in sl_pairs  # 1G
    assert (0.0, 2.0) in sl_pairs  # 1D
    assert (0.0, 0.0) in sl_pairs  # 1S


# ─── U^(k) unit tensor tests ──────────────────────────────

@pytest.mark.phase3
def test_uk0_diagonal_formula():
    """U^(0) diagonal should be n × √(2L+1) / √(2l+1)."""
    from multitorch.angular.rme import compute_uk_ls
    terms, parents, cfp = _get_terms_and_parents(2, 8)

    uk0 = compute_uk_ls(2, 8, 0, terms, parents, cfp)
    for i, t in enumerate(terms):
        expected = 8.0 * np.sqrt(2 * t.L + 1) / np.sqrt(5)
        assert abs(uk0[i, i] - expected) < 1e-4, (
            f"U^(0) for {t.label}: {uk0[i,i]:.6f} vs expected {expected:.6f}")


@pytest.mark.phase3
def test_uk0_is_diagonal():
    """U^(0) should be diagonal (k=0 is scalar)."""
    from multitorch.angular.rme import compute_uk_ls
    terms, parents, cfp = _get_terms_and_parents(2, 8)
    uk0 = compute_uk_ls(2, 8, 0, terms, parents, cfp)
    off_diag = uk0 - np.diag(np.diag(uk0))
    assert np.abs(off_diag).max() < 1e-10


@pytest.mark.phase3
def test_uk2_symmetry():
    """U^(k) should be symmetric."""
    from multitorch.angular.rme import compute_uk_ls
    terms, parents, cfp = _get_terms_and_parents(2, 8)
    for k in [0, 2, 4]:
        uk = compute_uk_ls(2, 8, k, terms, parents, cfp)
        assert np.allclose(uk, uk.T, atol=1e-10), f"U^({k}) not symmetric"


# ─── SHELL block validation against reference ─────────────

@pytest.fixture(scope="module")
def d8_shell_data():
    from multitorch.angular.rme import compute_uk_ls, compute_shell_blocks
    terms, parents, cfp = _get_terms_and_parents(2, 8)
    shell_ref, spin_ref = _load_ref_blocks(0)

    computed = {}
    for k in [0, 2, 4]:
        uk_ls = compute_uk_ls(2, 8, k, terms, parents, cfp)
        blocks = compute_shell_blocks(2, 8, k, terms, uk_ls)
        for (Jb, Jk), mat in blocks.items():
            computed[(k, Jb, Jk)] = mat

    return computed, shell_ref, spin_ref, terms


@pytest.mark.phase3
def test_shell_block_count(d8_shell_data):
    computed, shell_ref, _, _ = d8_shell_data
    # Should have all 36 SHELL1 blocks (k=0:5 + k=2:16 + k=4:15)
    assert len(computed) >= 36
    assert len(shell_ref) == 36


@pytest.mark.phase3
@pytest.mark.parametrize("k", [0, 2, 4])
def test_shell_blocks_match_reference(d8_shell_data, k):
    """All SHELL1 blocks for given k should match reference within 1e-5."""
    computed, shell_ref, _, _ = d8_shell_data

    max_err = 0.0
    count = 0
    for (rk, Jb, Jk), ref_mat in shell_ref.items():
        if rk != k:
            continue
        comp_key = (k, Jb, Jk)
        assert comp_key in computed, f"Missing block k={k}, J=({Jb},{Jk})"
        comp_mat = computed[comp_key]
        assert ref_mat.shape == comp_mat.shape
        err = np.abs(ref_mat - comp_mat).max()
        max_err = max(max_err, err)
        count += 1

    assert count > 0, f"No reference blocks for k={k}"
    assert max_err < 1e-5, f"k={k}: max error {max_err:.8f}"


# ─── SPIN block validation against reference ──────────────

@pytest.mark.phase3
def test_spin_blocks_match_reference(d8_shell_data):
    """All SPIN1 blocks should match reference within 1e-5."""
    from multitorch.angular.rme import compute_spin_blocks
    _, _, spin_ref, terms = d8_shell_data

    spin_computed = compute_spin_blocks(terms)

    max_err = 0.0
    count = 0
    for (Jb, Jk), ref_mat in spin_ref.items():
        assert (Jb, Jk) in spin_computed, f"Missing SPIN block J=({Jb},{Jk})"
        comp_mat = spin_computed[(Jb, Jk)]
        assert ref_mat.shape == comp_mat.shape
        err = np.abs(ref_mat - comp_mat).max()
        max_err = max(max_err, err)
        count += 1

    assert count == 12, f"Expected 12 SPIN1 blocks, got {count}"
    assert max_err < 1e-5, f"SPIN max error {max_err:.8f}"


# ─── rcg_cfp73 U^(k) table validation ─────────────────────

@pytest.mark.phase3
def test_cfp73_uk_matches_computed():
    """Pre-tabulated U^(k) from rcg_cfp73 should match computed U^(k)."""
    from multitorch.angular.rme import compute_uk_ls
    from multitorch.angular.cfp import get_uk_for_shell

    terms, parents, cfp = _get_terms_and_parents(2, 8)
    uk_block = get_uk_for_shell(2, 8)
    if uk_block is None:
        pytest.skip("rcg_cfp73 not found")

    for k in [0, 2, 4]:
        uk_computed = compute_uk_ls(2, 8, k, terms, parents, cfp)
        uk_table = uk_block.uk.get(k)
        if uk_table is None:
            continue
        # Table may have smaller shape (only nlt×nlt stored, skipping zeros)
        n = min(uk_computed.shape[0], uk_table.shape[0])
        m = min(uk_computed.shape[1], uk_table.shape[1])
        err = np.abs(uk_computed[:n, :m] - uk_table[:n, :m]).max()
        assert err < 1e-4, f"U^({k}) table vs computed: max error {err:.6f}"


# ─── ORBIT operator tests ─────────────────────────────────

@pytest.mark.phase3
def test_orbit_blocks_d8():
    """ORBIT blocks for d^8 must match analytical UNCPLA × sqrt(L(L+1)(2L+1)) values."""
    from multitorch.angular.rme import compute_orbit_blocks
    terms, _, _ = _get_terms_and_parents(2, 8)

    blocks = compute_orbit_blocks(terms)

    # d^8 terms: 3F(S=1,L=3), 3P(S=1,L=1), 1G(S=0,L=4), 1D(S=0,L=2), 1S(S=0,L=0)
    # ORBIT is rank-1 in J, diagonal in LS term.
    # Reference values from UNCPLA(L, S, J_bra, 1, L, J_ket) × sqrt(L(L+1)(2L+1)):
    assert len(blocks) == 12, f"Expected 12 ORBIT blocks for d^8, got {len(blocks)}"

    # Diagonal J blocks (selection rule ΔJ=0)
    ref_diag = {
        (1.0, 1.0): np.array([[1.224744871392]]),
        (2.0, 2.0): np.array([[7.302967433402, 0.0, 0.0],
                               [0.0, 2.738612787526, 0.0],
                               [0.0, 0.0, 5.477225575052]]),
        (3.0, 3.0): np.array([[8.401388774086]]),
        (4.0, 4.0): np.array([[10.062305898749, 0.0],
                               [0.0, 13.416407864999]]),
    }
    for (Jb, Jk), expected in ref_diag.items():
        assert (Jb, Jk) in blocks, f"Missing diagonal ORBIT block J={Jb}"
        np.testing.assert_allclose(blocks[(Jb, Jk)], expected, atol=1e-8,
            err_msg=f"ORBIT diagonal block J={Jb}")

    # Off-diagonal J blocks (ΔJ=±1): check antisymmetry ORBIT(J,J') = -ORBIT(J',J)^T
    for (Jb, Jk), mat in blocks.items():
        assert np.all(np.isfinite(mat)), f"ORBIT ({Jb},{Jk}) has non-finite values"
        if Jb != Jk and (Jk, Jb) in blocks:
            np.testing.assert_allclose(mat, -blocks[(Jk, Jb)].T, atol=1e-10,
                err_msg=f"ORBIT antisymmetry: ({Jb},{Jk}) vs ({Jk},{Jb})")


# ─── MULTIPOLE transition tests ───────────────────────────

@pytest.mark.phase3
def test_multipole_d8_p5d9_elementwise():
    """
    MULTIPOLE transition blocks (d^8 → p^5 d^9) should match reference
    element-wise after basis ordering and phase corrections.

    The two-shell coupled basis is sorted by (-S_total, -L_total) to match
    the Fortran convention, and a phase (-1)^{S+L+1} is applied per state.
    """
    import numpy as np
    from multitorch.angular.rme import compute_multipole_blocks
    from multitorch.angular.cfp import get_cfp_block
    from multitorch.io.read_rme import read_rme_rcg

    ref = read_rme_rcg(REFDATA / "nid8ct.rme_rcg")
    cfg0 = ref.configs[0]
    multi_ref = {}
    for b in cfg0.blocks:
        if b.operator == 'MULTIPOLE':
            Jb = float(int(b.bra_sym.replace('+','').replace('-','').replace('^','')))
            Jk = float(int(b.ket_sym.replace('+','').replace('-','').replace('^','')))
            multi_ref[(Jb, Jk)] = b.matrix.numpy()

    terms, parents, cfp = _get_terms_and_parents(2, 8)
    block_d8 = get_cfp_block(2, 8)

    computed = compute_multipole_blocks(
        l_gs=2, n_gs=8, l_core=1, n_core_gs=6,
        gs_terms=terms, gs_parents=parents, gs_cfp=block_d8.cfp,
    )

    assert set(computed.keys()) == set(multi_ref.keys()), \
        f"Block keys differ: computed={sorted(computed.keys())} vs ref={sorted(multi_ref.keys())}"

    max_err = 0.0
    for key in multi_ref:
        ref_mat = multi_ref[key]
        comp_mat = computed[key]
        assert ref_mat.shape == comp_mat.shape, \
            f"Shape mismatch for J={key}: {ref_mat.shape} vs {comp_mat.shape}"
        err = float(np.abs(ref_mat - comp_mat).max())
        max_err = max(max_err, err)

    assert max_err < 1e-5, f"Max element-wise error {max_err:.2e}"


# ─── High-level compute_all_shell_blocks test ─────────────

@pytest.mark.phase3
def test_compute_all_shell_blocks():
    from multitorch.angular.rme import compute_all_shell_blocks

    all_blocks = compute_all_shell_blocks(2, 8)
    # Should have blocks for k=0,2,4 at various J pairs
    k_values = set(key[0] for key in all_blocks)
    assert k_values == {0, 2, 4}
    assert len(all_blocks) >= 36
