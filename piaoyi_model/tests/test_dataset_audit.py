from __future__ import annotations

import numpy as np

from piaoyi_model.audit_dataset import longest_true_run


def test_longest_true_run_handles_partial_visibility_gaps() -> None:
    mask = np.asarray([False, True, True, False, True, True, True, False])
    assert longest_true_run(mask) == 3
    assert longest_true_run(np.zeros(5, dtype=np.bool_)) == 0

