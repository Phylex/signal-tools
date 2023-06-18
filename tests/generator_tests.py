from typing import List, Tuple, Union
import pytest
from numpy._typing import NDArray
from itertools import chain
from signal_tools.generators import rngen


@pytest.mark.parametrize("shape, range_, chunk_size, samples",
                         [
                             ((1,), (0, 10), 10, 100),
                             ((2, 3), (10, 20), 10, None),
                          ]
                         )
def test_rn_gen(shape: Tuple[int], range_: Tuple[float, float], chunk_size: int, samples: Union[int, None]):
    if samples is not None:
        rnums = list(rngen(shape, range_, chunk_size, samples))
        assert len(rnums) == samples
    else:
        rnumgen = rngen(shape, range_, chunk_size, samples)
        i = 0
        rnums: List[NDArray] = []
        while i < 100:
            rnums.append(next(rnumgen))
            i += 1
        assert len(rnums) == 100
    flatnums = list(chain(*[rn.flatten() for rn in rnums]))
    print(flatnums)
    assert max(flatnums) <= max(range_)
    assert min(flatnums) >= min(range_)
    for rn in rnums:
        assert rn.shape == shape
