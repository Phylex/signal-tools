"""
Test the various mo
"""
import pytest
import numpy as np
from signals_systems_filters.filters import g_h_filter


@pytest.mark.parametrize(
        "g, h, mom_init, state_init, data, truth",
        [(0.3, 0.3, 0, 0, np.random.randn(300), lambda _: 0),
         (0.3, 0.3, 0, 0, np.arange(300) + np.random.rand(300), lambda x: x)],
        )
def test_gh_filter(g, h, mom_init, state_init, data, truth):
    pass
