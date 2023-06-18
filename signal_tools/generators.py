from itertools import chain, repeat
from typing import Generator, Tuple, Union
import numpy as np
from numpy._typing import NDArray


def rngen(shape: Tuple[int],
          range_: Tuple[float, float],
          chunk_size: int,
          samples: Union[int, None]) -> Generator[NDArray[float], None, None]:
    """
    Generator of <sample> number of uniform random number Tensors in <range_>

    Generator generates random tensors in batches of <chunk_size>
    and yields them one after the other until <samples> have been generated
    then terminates.
    """
    delta = max(range_) - min(range_)
    offset = min(range_)
    if samples is not None:
        full_chunks = samples // chunk_size
        last_chunk_shape = [samples - (full_chunks * chunk_size)] + list(shape)
        chunk_shapes = chain(repeat([chunk_size] + list(shape), full_chunks),
                             [last_chunk_shape])
    else:
        chunk_shapes = repeat([chunk_size] + list(shape))
    for chunk_shape in chunk_shapes:
        rnums = np.random.rand(*chunk_shape) * delta + offset
        for rt in rnums:
            yield rt
