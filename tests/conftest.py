"""Shared fixtures for Rank-GRPO parity tests."""

import os
import pickle
import sys
import tempfile
from pathlib import Path

import pytest

# -- path setup: ensure the repo root (parent of verl_gr/) is on sys.path ----
_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "verl_gr").is_dir():
    _p = _p.parent
_REPO_ROOT = str(_p) if (_p / "verl_gr").is_dir() else None
if _REPO_ROOT is not None and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Ground-truth catalog used by test_rankgrpo_parity
GT_CATALOG = frozenset({
    ("The Shawshank Redemption", 1994),
    ("The Godfather", 1972),
    ("Pulp Fiction", 1994),
    ("The Dark Knight", 2008),
    ("Schindler's List", 1993),
    ("Forrest Gump", 1994),
    ("Inception", 2010),
    ("Fight Club", 1999),
    ("The Matrix", 1999),
    ("Goodfellas", 1990),
})


def pytest_addoption(parser):
    parser.addoption("--regenerate", action="store_true", help="Regenerate golden values")
    parser.addoption("--verl-path", type=str, default=None, help="Path to verl library")
    parser.addoption("--skip-model-tests", action="store_true", help="Skip tests requiring HF model")


@pytest.fixture(scope="session")
def regenerate(request):
    return request.config.getoption("--regenerate", default=False)


@pytest.fixture(scope="session")
def catalog_path():
    """Build a temporary GT catalog pickle file for the session."""
    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="rankgrpo_test_catalog_")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(list(GT_CATALOG), f)
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def model_path():
    return os.environ.get("RANKGRPO_TEST_MODEL_PATH", "")


@pytest.fixture(scope="session")
def have_verl(request):
    """Ensure verl is importable, adding --verl-path if provided."""
    verl_path = request.config.getoption("--verl-path", default=None)
    if verl_path:
        if verl_path not in sys.path:
            sys.path.insert(0, verl_path)
    env_path = os.environ.get("VERL_LIB_PATH", "")
    if env_path:
        if env_path not in sys.path:
            sys.path.insert(0, env_path)
    try:
        import verl  # noqa: F401
        return True
    except ImportError:
        return False
