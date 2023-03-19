from itertools import accumulate, repeat, starmap, tee
from operator import add, mul
from typing import Iterator, Iterable
from numbers import Number


def _delayed_subtractor(delay: int, input: Iterable[Number]) -> Iterator[Number]:
    delay_fifo = []
    for val in input:
        if len(delay_fifo) < delay:
            delay_fifo.append(val)
            yield val
        else:
            delayed_val = delay_fifo.pop(0)
            delay_fifo.append(val)
            yield val - delayed_val


def trapezoid_filter(rise_time: int,
                     trapezoid_len: int,
                     decay_compensation: float,
                     input: Iterable[Number]) -> Iterator[Number]:
    """
    Implements the trapezoid filter as in iterator over values
    """
    in_1, in_2 = tee(input, 2)
    return accumulate(
        starmap(
            add,
            zip(
                starmap(
                    mul,
                    zip(
                        _delayed_subtractor(trapezoid_len,
                                           _delayed_subtractor(rise_time, in_1)),
                        repeat(decay_compensation))),
                accumulate(
                    _delayed_subtractor(trapezoid_len,
                                       _delayed_subtractor(rise_time, in_2)))
            )
        )
    )


def g_h_filter(data: Iterable[float],
               initial_momentum: float,
               initial_state: float,
               h: float,
               g: float,
               timestep: float) -> Iterator[float]:
    """
    A g-h filter with fixed predetermined g and h

    A filter that uses a 'state measument' and a numerically calculated
    'state velocity' to track a system. This filter returns an iterator
    and expects the measurements to be an Iterable

    :param data: The measurements of the system state.
    :type data: Iterable[float], required
    :param initail_velo: Initial guess of the 'velocity' of the system.
        The definition of 'velocity' here means rate of change of system
        state represented by the tracked variable (the system state).
    :type initial_momentum: float, required
    :param initial_state: The initial system state at the start of the
        observations
    :type initial_state: float, required
    :param h: Set the impact of an observed 'velocity change' on the
        estimate of the system velocity. A large value makes a new
        measurement of system velocity. An agile system (corresponding to a
        large value) will 'react quickly' to a change in observed velocity,
        and a sluggish (small values) system will not. If we are modeling a
        sluggish system it is unlikely fow it to change system velocity even
        if the obesrvables seem to indicate a large change in the systems
        trajectory.
    :type h: float, required
    :param g: Determines how much the measurement influences
        the output of the estimated system state. Large values correspond to
        a high confidence in the measured value low values correspond to low
        confidence in the accuracy of the observables.
    :type g: float, required
    :param timestep: The time difference between two measurments, and
        subsequently the time for witch the state of the system is predicted
    :type timestep: float, required
    """
    momentum = initial_momentum
    state = initial_state

    for measurement in data:
        # prediction step
        prediction = state + timestep * momentum
        # measurment step
        residual = measurement - prediction

        # update the weights used in generating the next prediction
        momentum += h * residual
        state = prediction + g * residual
        yield state
