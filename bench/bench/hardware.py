"""Hardware and version metadata captured once per benchmark session.

Every cell record embeds this blob so downstream post-processing can
distinguish runs done on different machines / tool versions without
relying on external state.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from bench.config import MULTITORCH_ROOT, PYCTM_ROOT, PYTTMULT_ROOT, TTMULT_ROOT


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _git_sha(repo_root: Path) -> Optional[str]:
    if not (repo_root / ".git").exists():
        return None
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return None


def _sha256(path: Path, limit_bytes: int = 8 * 1024 * 1024) -> Optional[str]:
    """Return sha256 of a file; limit_bytes caps bytes read for very large files."""
    try:
        h = hashlib.sha256()
        read = 0
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
                read += len(chunk)
                if read >= limit_bytes:
                    break
        return h.hexdigest()
    except Exception:
        return None


def collect_hardware() -> Dict:
    """Capture CPU/GPU/RAM + library versions + git/binary fingerprints."""
    info: Dict = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
    }

    # psutil for CPU/RAM detail
    try:
        import psutil
        info["cpu_logical"] = psutil.cpu_count(logical=True)
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        info["cpu_logical"] = os.cpu_count()
        info["cpu_physical"] = None
        info["ram_gb"] = None

    # torch + cuda
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        info["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False

    # numpy / scipy versions
    for mod_name in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            mod = __import__(mod_name)
            info[mod_name] = getattr(mod, "__version__", None)
        except ImportError:
            info[mod_name] = None

    # Git SHAs for the four source trees
    info["git"] = {
        "multitorch": _git_sha(MULTITORCH_ROOT),
        "pyctm": _git_sha(PYCTM_ROOT),
        "pyttmult": _git_sha(PYTTMULT_ROOT),
    }

    # Fortran binary fingerprints
    bin_root = TTMULT_ROOT / "bin"
    bin_info: Dict[str, Dict] = {}
    for name in ("ttrcg", "ttrac", "ttban", "ttban_exact", "rcn31", "rcn2",
                 "rcg_cfp72", "rcg_cfp73", "rcg_cfp74"):
        p = bin_root / name
        if p.exists():
            st = p.stat()
            bin_info[name] = {
                "size_bytes": st.st_size,
                "mtime": int(st.st_mtime),
                "sha256_prefix": (_sha256(p) or "")[:16],
            }
    info["fortran_binaries"] = bin_info
    info["fortran_bin_root"] = str(bin_root)

    return info


def hardware_hash(hw: Dict) -> str:
    """Short hash over the fields that *define* a reproducible run.

    Used for detecting "same machine, same versions" in --resume. Changes
    in OS minor version / RAM bump are tolerated; changes in torch /
    cuda / binary hashes are not.
    """
    keys = (
        ("platform",),
        ("torch",),
        ("cuda_version",),
        ("gpu_name",),
        ("numpy",),
        ("git", "multitorch"),
        ("git", "pyctm"),
        ("git", "pyttmult"),
    )
    parts = []
    for path in keys:
        node: object = hw
        for key in path:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                node = None
                break
        parts.append(f"{'.'.join(path)}={node}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
