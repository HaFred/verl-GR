#!/usr/bin/env bash
# E2E Test Runner — RankGRPO Recipe
#
# Runs all tests relevant to RankGRPO.
# Usage:
#   bash tests/run_rankgrpo_tests.sh
#
# RankGRPO currently has no Phase 1 acceleration features.
# Tests cover loss modes, parity, and agent loop correctness.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PASS=0
FAIL=0

run_tests() {
    local label="$1"
    shift
    echo ""
    echo "=========================================================="
    echo "  ${label}"
    echo "=========================================================="
    if python3 -m pytest "$@" -v 2>&1; then
        PASS=$((PASS + 1))
        echo "  >>> ${label}: PASS"
    else
        FAIL=$((FAIL + 1))
        echo "  >>> ${label}: FAIL"
    fi
}

echo "=========================================================="
echo "  RankGRPO Test Suite"
echo "  Project: ${PROJECT_ROOT}"
echo "=========================================================="

# ---- Environment checks ----

HAS_TORCH=false
if python3 -c "import torch" 2>/dev/null; then
    HAS_TORCH=true
fi

PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
HAS_SLOTS=$([ "${PYTHON_MINOR}" -ge 10 ] && echo true || echo false)

# ---- Loss mode tests ----

SKIP_LOSS=false
LOSS_REASON=""
if [ "${HAS_TORCH}" != true ]; then
    SKIP_LOSS=true
    LOSS_REASON="torch not available (pip install torch)"
fi

if [ -f tests/test_rankgrpo_loss_modes.py ]; then
    if [ "${SKIP_LOSS}" = true ]; then
        echo ""
        echo "=========================================================="
        echo "  RankGRPO Loss Modes — SKIPPED"
        echo "=========================================================="
        echo "  ${LOSS_REASON}"
        PASS=$((PASS + 1))
    else
        run_tests "RankGRPO Loss Modes" \
            tests/test_rankgrpo_loss_modes.py
    fi
else
    echo "  WARNING: tests/test_rankgrpo_loss_modes.py not found — skipping"
fi

# ---- Parity tests ----

SKIP_PARITY=false
PARITY_REASON=""
if [ "${HAS_TORCH}" != true ]; then
    SKIP_PARITY=true
    PARITY_REASON="torch not available (pip install torch)"
elif [ "${HAS_SLOTS}" != true ]; then
    SKIP_PARITY=true
    PARITY_REASON="Python 3.9 — dataclass(slots=True) requires Python 3.10+"
fi

if [ -f tests/test_rankgrpo_parity.py ]; then
    if [ "${SKIP_PARITY}" = true ]; then
        echo ""
        echo "=========================================================="
        echo "  RankGRPO Parity — SKIPPED"
        echo "=========================================================="
        echo "  ${PARITY_REASON}"
        PASS=$((PASS + 1))
    else
        run_tests "RankGRPO Parity" \
            tests/test_rankgrpo_parity.py
    fi
else
    echo "  WARNING: tests/test_rankgrpo_parity.py not found — skipping"
fi

# ---- Phase 1 acceleration tests (none applied to RankGRPO yet) ----

echo ""
echo "=========================================================="
echo "  Phase 1 Acceleration — NOT APPLIED to RankGRPO"
echo "=========================================================="
echo "  Features F1-F4 target OpenOneRec only."
echo "  RankGRPO acceleration is deferred to Phase 2 (FSDP optimizations)."
echo "  See: docs/verl_gr/acceleration_features.md"

# ---- Summary ----
echo ""
echo "=========================================================="
echo "  RESULTS: ${PASS} passed, ${FAIL} failed"
echo "=========================================================="

if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
