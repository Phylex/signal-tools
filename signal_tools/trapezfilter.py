from collections.abc import Iterable
from itertools import accumulate, repeat, starmap, tee
from numbers import Real
from operator import add, mul


def delayed_subtractor(delay: int, input: Iterable[Real]) -> Iterable[Real]:
    delay_fifo = []
    for val in input:
        if len(delay_fifo) < delay:
            delay_fifo.append(val)
            yield val
        else:
            delayed_val = delay_fifo.pop(0)
            delay_fifo.append(val)
            yield val - delayed_val


def trapezoid_filter(k: int, l: int, m: int, input: Iterable[Real]) -> Iterable[Real]:
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
                        delayed_subtractor(l,
                                           delayed_subtractor(k, in_1)),
                        repeat(m))),
                accumulate(
                    delayed_subtractor(l,
                                       delayed_subtractor(k, in_2)))
            )
        )
    )


class TrapezoidFilter():
    def __init__(self, k: int, l: int, m: int, input: Iterable[Real]):
        self.dkl = DelayedSubtractor(l, DelayedSubtractor(k, input))
        self.k = k
        self.l = l
        self.m = m
        self.m = m

    def __iter__(self) -> Iterable[Real]:
        return self.acc_2.shift_in((dkl*self.m)+self.acc_1.shift_in(dkl))
