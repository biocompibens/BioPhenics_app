from numpy import inf
from pandas import DataFrame


class AtomicInterval:
    """
    This class represents an atomic interval.
    An atomic interval is a single interval, with a lower and upper bounds,
    and two (closed or open) boundaries.
    """

    __slots__ = ('_lower', '_upper')

    def __init__(self, lower, upper):
        """
        Create an atomic interval.
        If a bound is set to infinity (regardless of its sign),
        the corresponding boundary will be exclusive.
        :param lower: lower bound value.
        :param upper: upper bound value.
        """
        self._lower = lower
        self._upper = upper

        if self.is_empty():
            self._lower = inf
            self._upper = -inf

    @property
    def lower(self):
        """
        Lower bound value.
        """
        return self._lower

    @property
    def upper(self):
        """
        Upper bound value.
        """
        return self._upper

    def is_empty(self):
        """
        Test interval emptiness.
        :return: True if interval is empty, False otherwise.
        """
        return self._lower >= self._upper

    def overlaps(self, other):
        """
        Test if intervals have any overlapping value.
        :param other: an atomic interval.
        :return True if intervals overlap, False otherwise.
        """
        if not isinstance(other, AtomicInterval):
            raise TypeError('Only AtomicInterval instances are supported.')

        if self._lower > other.lower:
            first, second = other, self
        else:
            first, second = self, other

        return first._upper >= second._lower

    def __and__(self, other):
        if isinstance(other, AtomicInterval):
            if self._lower == other._lower:
                lower = self._lower
            else:
                lower = max(self._lower, other._lower)

            if self._upper == other._upper:
                upper = self._upper
            else:
                upper = min(self._upper, other._upper)

            if lower <= upper:
                return AtomicInterval(lower, upper)
            return AtomicInterval(lower, lower)
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, AtomicInterval):
            if self.overlaps(other):
                if self._lower == other._lower:
                    lower = self._lower
                else:
                    lower = min(self._lower, other._lower)

                if self._upper == other._upper:
                    upper = self._upper
                else:
                    upper = max(self._upper, other._upper)

                return AtomicInterval(lower, upper)
            return Interval(self, other)
        return NotImplemented

    def __contains__(self, item):
        if isinstance(item, AtomicInterval):
            left = item._lower >= self._lower
            right = item._upper <= self._upper
            return left and right
        elif isinstance(item, Interval):
            for interval in item:
                if interval not in self:
                    return False
            return True
        else:
            left = item >= self._lower
            right = item <= self._upper
            return left and right

    def __invert__(self):
        return Interval(
            AtomicInterval(-inf, self._lower),
            AtomicInterval(self._upper, inf)
        )

    def __sub__(self, other):
        if isinstance(other, AtomicInterval):
            return self & ~other
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, AtomicInterval):
            return (
                    self._lower == other._lower and
                    self._upper == other._upper
            )
        return NotImplemented

    def __ne__(self, other):
        return not self == other  # Required for Python 2

    def __repr__(self):
        return f"[{self._lower};{self._upper}]"

    def isin(self, serie):
        """
        check element-wise if serie is in interval

        Parameters
        ----------
        serie

        Returns
        -------

        """
        if isinstance(serie, DataFrame):
            return DataFrame({col: self.isin(serie[col]) for col in serie},
                             index=serie.index)
        try:
            return type(serie)([elem in self for elem in serie],
                               index=serie.index)
        except TypeError:
            return type(serie)([elem in self for elem in serie])
        except BaseException:
            return [elem in self for elem in serie]


class Interval:
    """
    This class represents an interval.
    An interval is an (automatically simplified) union of atomic intervals.
    It can be created either by passing (atomic) intervals, or by using
    one of the helpers
    provided in this module (open(..), closed(..), etc).
    Unless explicitly specified, all operations on an Interval instance
    return Interval instances.
    """

    __slots__ = ('_intervals',)

    def __init__(self, *intervals):
        """
        Create an interval from a list of (atomic or not) intervals.
        :param intervals: a list of (atomic or not) intervals.
        """
        self._intervals = list()

        for interval in intervals:
            if isinstance(interval, Interval):
                self._intervals.extend(interval)
            elif isinstance(interval, AtomicInterval):
                if not interval.is_empty():
                    self._intervals.append(interval)
            else:
                raise TypeError('Parameters must be Interval '
                                'or AtomicInterval instances')

        if len(self._intervals) == 0:
            # So we have at least one (empty) interval
            self._intervals.append(AtomicInterval(inf, -inf))
        else:
            # Sort intervals by lower bound
            self._intervals.sort(key=lambda x: x.lower)

            i = 0
            # Attempt to merge consecutive intervals
            while i < len(self._intervals) - 1:
                current = self._intervals[i]
                successor = self._intervals[i + 1]

                if current.overlaps(successor):
                    interval = current | successor
                    self._intervals.pop(i)
                    self._intervals.pop(i)
                    self._intervals.insert(i, interval)
                else:
                    i = i + 1

    def is_atomic(self):
        """
        Test interval atomicity. An interval is atomic if it is composed
         of a single atomic interval.

        :return: True if this interval is atomic, False otherwise.

        """
        return len(self._intervals) == 1

    def overlaps(self, other):
        """
        Test if intervals have any overlapping value.
        If 'permissive' is set to True (default is False),
        then [1, 2) and [2, 3] are considered as having
        an overlap on value 2 (but not [1, 2) and (2, 3]).
        :param other: an interval or atomic interval.
        :return True if intervals overlap, False otherwise.
        """
        if isinstance(other, AtomicInterval):
            for interval in self._intervals:
                if interval.overlaps(other):
                    return True
            return False
        elif isinstance(other, Interval):
            for o_interval in other._intervals:
                if self.overlaps(o_interval):
                    return True
            return False
        else:
            raise TypeError(f'Unsupported type {type(other)} for {other}')

    def __len__(self):
        if self._intervals[0].is_empty():
            return 0
        return len(self._intervals)

    def __iter__(self):
        if self._intervals[0].is_empty():
            return iter([])
        return iter(self._intervals)

    def __and__(self, other):
        if isinstance(other, (AtomicInterval, Interval)):
            if isinstance(other, AtomicInterval):
                intervals = [other]
            else:
                intervals = list(other._intervals)
            new_intervals = []
            for interval in self._intervals:
                for o_interval in intervals:
                    new_intervals.append(interval & o_interval)
            return Interval(*new_intervals)
        return NotImplemented

    def __contains__(self, item):
        if isinstance(item, Interval):
            for o_interval in item._intervals:
                if o_interval not in self:
                    return False
            return True
        elif isinstance(item, AtomicInterval):
            for interval in self._intervals:
                if item in interval:
                    return True
            return False
        else:
            for interval in self._intervals:
                if item in interval:
                    return True
            return False

    def __repr__(self):
        return ' | '.join(repr(i) for i in self._intervals)
