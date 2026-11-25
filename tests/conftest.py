"""Suite-wide setup.

**The tests run on the CPU, always.**  JAX takes the GPU by default and
preallocates most of its memory, so the suite is only reliable when nothing else
is using the device -- and something usually is.  With a Step 3 grid running, the
full suite failed nine tests with
``RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY``, including Step 1's
``test_two_class_filter_reduces_to_one_class_when_a_class_is_empty``, which has
nothing to do with the grid.  A test that fails because of another process is
worse than a slow test: it trains the reader to dismiss red.

Pinning here is also free on merit.  Step 1 measured the GPU **4.8x slower** than
the CPU at this project's dominant shape -- a sequential scan over 41-vector
bodies is latency-bound, so per-step launch overhead dominates -- and
``check_a_dynamics.py`` already pins itself to the CPU for the same reason.

This has to happen before anything imports JAX, which is why it is in
``conftest.py`` at the suite root rather than in a fixture: by the time a fixture
runs, the test module has already been imported and JAX has already chosen a
backend.  Set ``GN_TEST_DEVICE=gpu`` to override, for deliberately measuring the
device rather than using it by accident.
"""

from __future__ import annotations

import os

if os.environ.get("GN_TEST_DEVICE", "cpu") == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
