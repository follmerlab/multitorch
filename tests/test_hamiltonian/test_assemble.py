"""
Tests for Hamiltonian assembly from .rme_rcg, .rme_rac, and .ban files.

Validates:
  1. Ban file parser: correct NCONF, XHAM, XMIX, TRAN, TRIADS
  2. COWAN store: correct number of sections and matrices per section
  3. Ground state eigenvalues match nid8ct.ban_out for all triads (atol=0.001 Ry)
  4. Total transition intensities match nid8ct.ban_out (atol=0.001)
  5. Configuration weights in ground state match reference
"""
import pytest
import torch
from pathlib import Path

REFDATA = Path(__file__).parent.parent / "reference_data" / "nid8ct"


@pytest.mark.phase2
def test_read_ban():
    from multitorch.io.read_ban import read_ban
    ban = read_ban(REFDATA / "nid8ct.ban")
    assert ban.nconf_gs == 2
    assert ban.nconf_fs == 2
    assert ban.eg[2] == 5.0
    assert ban.ef[2] == 4.0
    assert len(ban.xham) == 1
    assert ban.xham[0].values == [1.0, 1.0, 0.0, 0.1]
    assert len(ban.xmix) == 1
    assert ban.xmix[0].values == [2.0, 2.0, 1.0, 1.0]
    assert ban.tran == [(1, 1), (2, 2)]
    assert len(ban.triads) == 13


@pytest.mark.phase2
def test_cowan_store_sections():
    from multitorch.io.read_rme import read_cowan_store
    sections = read_cowan_store(REFDATA / "nid8ct.rme_rcg")
    assert len(sections) == 4
    assert len(sections[0]) == 22   # conf 1 dipole
    assert len(sections[1]) == 24   # conf 2 dipole
    assert len(sections[2]) == 167  # ground mixing
    assert len(sections[3]) == 142  # excited mixing


@pytest.mark.phase2
def test_rme_rac_full_blocks():
    from multitorch.io.read_rme import read_rme_rac_full
    rac = read_rme_rac_full(REFDATA / "nid8ct.rme_rac")
    assert len(rac.blocks) == 226
    # First TRANSI block
    b0 = rac.blocks[0]
    assert b0.kind == 'TRANSI'
    assert b0.bra_sym == '0+'
    assert b0.n_bra == 9
    assert b0.n_ket == 15
    assert len(b0.add_entries) == 11


# ─── Ground state eigenvalue tests ───────────────────────────

REF_EG0 = {
    ('0+', '1-', '1-'): -2.190453,
    ('^0+', '1-', '1-'): -2.164810,
    ('1+', '1-', '0-'): -3.450601,
    ('1+', '1-', '^0-'): -3.450601,
    ('1+', '1-', '2-'): -3.450601,
    ('1+', '1-', '^2-'): -3.450601,
    ('2+', '1-', '1-'): -2.160916,
    ('^2+', '1-', '1-'): -3.450531,
    ('0+', '^0-', '^0-'): -2.190453,
    ('^0+', '^0-', '0-'): -2.164810,
    ('1+', '^0-', '1-'): -3.450601,
    ('2+', '^0-', '^2-'): -2.160916,
    ('^2+', '^0-', '2-'): -3.450531,
}

REF_TOTAL_INTENSITY = {
    ('0+', '1-', '1-'): 0.27102,
    ('^0+', '1-', '1-'): 0.27069,
    ('1+', '1-', '0-'): 0.11438,
}


@pytest.fixture(scope="module")
def nid8ct_assembled():
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize
    return assemble_and_diagonalize(
        REFDATA / "nid8ct.rme_rcg",
        REFDATA / "nid8ct.rme_rac",
        REFDATA / "nid8ct.ban",
    )


@pytest.mark.phase2
def test_all_triads_processed(nid8ct_assembled):
    assert len(nid8ct_assembled.triads) == 13


@pytest.mark.phase2
@pytest.mark.parametrize("triad,ref_Eg0", list(REF_EG0.items()))
def test_ground_state_eigenvalue(nid8ct_assembled, triad, ref_Eg0):
    """Ground state eigenvalue Eg0 should match reference within 0.001 Ry."""
    gs, act, fs = triad
    t = nid8ct_assembled.get_triad(gs, act, fs)
    assert t is not None, f"Triad ({gs}, {act}, {fs}) not found"
    Eg0 = float(t.Eg[0])
    assert abs(Eg0 - ref_Eg0) < 0.001, (
        f"Eg0 for ({gs},{act},{fs}): {Eg0:.6f} vs ref {ref_Eg0:.6f}"
    )


@pytest.mark.phase2
@pytest.mark.parametrize("triad,ref_I", list(REF_TOTAL_INTENSITY.items()))
def test_total_transition_intensity(nid8ct_assembled, triad, ref_I):
    """Total transition intensity from ground state should match reference."""
    gs, act, fs = triad
    t = nid8ct_assembled.get_triad(gs, act, fs)
    assert t is not None
    M2 = (t.T[0, :] ** 2).sum().item()
    assert abs(M2 - ref_I) < 0.001, (
        f"Total I for ({gs},{act},{fs}): {M2:.5f} vs ref {ref_I:.5f}"
    )


@pytest.mark.phase2
def test_ground_conf_weights(nid8ct_assembled):
    """Configuration weights in ground state should match reference."""
    # Reference from ban_out: triad (0+,1-,1-):
    # Weight of configurations 1,2,3: 0.91158 0.08842 0.00000
    t = nid8ct_assembled.get_triad('0+', '1-', '1-')
    assert t is not None

    # Compute weight of each config in the ground state eigenvector
    gs_vec = t.Ug[:, 0]  # ground state eigenvector
    w1 = (gs_vec[t.gs_conf_labels == 1] ** 2).sum().item()
    w2 = (gs_vec[t.gs_conf_labels == 2] ** 2).sum().item()

    assert abs(w1 - 0.91158) < 0.01, f"Config 1 weight: {w1:.5f} vs ref 0.91158"
    assert abs(w2 - 0.08842) < 0.01, f"Config 2 weight: {w2:.5f} vs ref 0.08842"


@pytest.mark.phase2
def test_ground_state_dimensions(nid8ct_assembled):
    """Ground state dimensions should be correct for 2-config CT."""
    t = nid8ct_assembled.get_triad('0+', '1-', '1-')
    assert t.n_gs == 22  # 9 + 13
    assert t.gs_conf_sizes == [9, 13]

    t2 = nid8ct_assembled.get_triad('1+', '1-', '0-')
    assert t2.n_gs == 35  # 10 + 25
    assert t2.gs_conf_sizes == [10, 25]


@pytest.mark.phase2
def test_excited_state_dimensions(nid8ct_assembled):
    """Final state should include both configurations."""
    t = nid8ct_assembled.get_triad('0+', '1-', '1-')
    assert t.n_fs == 30  # 15 + 15
    assert t.fs_conf_sizes == [15, 15]


@pytest.mark.phase2
def test_in_memory_entry_point_byte_equivalent(nid8ct_assembled):
    """
    Phase 5 contract: ``assemble_and_diagonalize_in_memory`` must produce
    the same BanResult as the file-based wrapper when fed the parser
    outputs directly. This is the regression that locks in the C1
    refactor — if anyone removes the wrapper or breaks the in-memory
    entry point, this test catches it.
    """
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.io.read_rme import read_cowan_store, read_rme_rac_full
    from multitorch.io.read_ban import read_ban

    cowan = read_cowan_store(REFDATA / "nid8ct.rme_rcg")
    rac = read_rme_rac_full(REFDATA / "nid8ct.rme_rac")
    ban = read_ban(REFDATA / "nid8ct.ban")
    in_mem = assemble_and_diagonalize_in_memory(cowan, rac, ban)

    assert len(in_mem.triads) == len(nid8ct_assembled.triads)
    for t_mem, t_file in zip(in_mem.triads, nid8ct_assembled.triads):
        assert t_mem.gs_sym == t_file.gs_sym
        assert t_mem.act_sym == t_file.act_sym
        assert t_mem.fs_sym == t_file.fs_sym
        assert torch.allclose(t_mem.Eg, t_file.Eg, atol=1e-12)
        assert torch.allclose(t_mem.Ef, t_file.Ef, atol=1e-12)
        assert torch.allclose(t_mem.T, t_file.T, atol=1e-12)
