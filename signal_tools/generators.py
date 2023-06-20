from itertools import chain, repeat
from typing import Callable, Generator, Tuple, Union, Iterator
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

distributions = {"uniform": np.random.rand,
                 "normal": np.random.normal,
                 }


def pulse_gen(rate: float,
              decay_const: float,
              pulse_height_dist: Callable,
              dist_params: Tuple[float]):
    """
    Generate a stream of pulses with the pulse heights distributions following the
    <pulse_height_dist> function
    """

    # iterator for generating a single pulse
    def pulse(pulse_height: float, decay_const: float) -> Iterator[float]:
        return map(lambda i: np.exp(-1/decay_const * i) * pulse_height,
                   range(int(decay_const * 6)))

    # the pulses that are not yet 0 need to be tracked so that
    # pileup can be simulated
    active_pulses = []

    # loop to generate the data stream
    while True:
        if np.random.rand() > 1/rate:
            active_pulses.append(
                    pulse(pulse_height_dist(*dist_params), decay_const))
        sig_val: float = 0
        finished_pulses = []
        for pulse in active_pulses:
            try:
                sig_val += next(pulse)
            except StopIteration:
                finished_pulses.append(pulse)
        # remove the pulses that have been exhausted
        for fp in finished_pulses:
            active_pulses.remove(fp)
        yield sig_val
