from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


def roundx(a: ArrayLike, x: Union[int, float] = 1) -> NDArray:
    """Rounds values to the closest multiple of **x**.

    Args:
        a (ArrayLike): numeric array
        x (Union[int, float], optional): int or float. Defaults to 1.

    Returns:
        NDArray: array of rounded values

    Examples:
        All values of the array are rounded to the closes multiple of **x**.

        >>> a = np.array([15, 180, 45])
        >>> helper.roundx(a, 17)
        array([ 17., 187.,  51.])

        This does also work with floats.

        >>> a = np.array([10.0, 4.5, 3.5])
        >>> helper.roundx(a, 0.7)
        array([9.8, 4.2, 3.5])
    """
    a = np.asarray(a)
    return np.around(x * np.around(a / x), number_precision((x,)))


def number_precision(a: ArrayLike, max: int = 16) -> int:
    """Returns the decimal index to which rounding does not change the values.
    Float precision is capped at **max** value. This can be used to count
    decimals of floats, or trailing zeros of integers.

    Args:
        a (ArrayLike): numeric array
        max (int, optional): maximal inspected float precision. Defaults to 16.

    Returns:
        int: decimal index, can be negative

    Examples:
        The function can be used to get the maximal number of decimals that is
        required to represent all floats in **a** without loss.

        >>> a = np.array([1.5, 1.830, 45.0])
        >>> helper.number_precision(a)
        2

        Values with traling zeros have a negative precesion.

        >>> a = np.array([150, 1800, 450])
        >>> helper.number_precision(a)
        -1

        In the example above, rounding **a** with decimals -1 has no impact.
    """
    a = np.asarray(a)[~np.isnan(np.asarray(a))]
    d = -int(np.ceil(np.log10(np.abs(np.amax(a)))))
    while any(a != np.around(a, d)) and d < max:
        d += 1
    return d


def gcd(a: ArrayLike) -> Union[int, float]:
    """A float generalisation of the greatest common divisor.

    Args:
        a (ArrayLike): numeric array

    Returns:
        Union[int, float]: greatest common divisor of the array

    Examples:
        The function generalises the greatest common divisor to floats.

        >>> a = np.array([10.0, 4.5, 3.5])
        >>> helper.gcd(a)
        0.5

        However, it does also work with int.

        >>> a = np.array([15, 180, 45])
        >>> helper.gcd(a)
        15
    """
    a = np.asarray(a)
    n = number_precision(a)
    int_repr = np.around(a * (10 ** n)).astype(np.int_)
    gcd_float = np.around(np.gcd.reduce(int_repr) / (10 ** n), decimals=n)
    return a.dtype.type(gcd_float).item()


def datetime_precision(a: ArrayLike) -> str:
    """Returns the highest datetime precision present in the data. The precision
    is encoded as character *ns*, *ms*, *s*, *m*, *h*, *D*, *M*, and *Y*.

    Args:
        a (ArrayLike): datetime array

    Returns:
        str: datetime precision of the array

    Examples:
        The function does not derive the precision from the data type, but from
        the actual values.

        >>> a = np.array(['2012-03', '2012-05-12T12:00'], dtype=np.datetime64)
        array(['2012-03-01T00:00', '2012-05-12T12:00', dtype='datetime64[m]'])
        >>> datetime_precision(a)
        'h'
    """
    a = np.asarray(a)
    freq = "ns"
    for f in ["ms", "s", "m", "h", "D", "M", "Y"]:
        if any(a != a.astype(f"datetime64[{f}]")):
            break
        else:
            freq = f
    return freq


def interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> NDArray:
    """One-dimensional linear interpolation for monotonically increasing points.
    Note: numpy.interp does not work with datetime values.

    Args:
        x (ArrayLike): x-coordinates at which to evaluate the interpolation.
        xp (ArrayLike): x-coordinates of the data points.
        fp (ArrayLike): y-coordinates of the data points.

    Returns:
        NDArray: array of values at interpolated y-coordinates.

    Examples:
        With int and float arrays this function works like numpy.interp.

        >>> xp = np.array([0, 10, 100])
        >>> fp = np.array([4, 7, 25])
        >>> x = np.array([6, 17])
        >>> helper.interp(x, xp, fp)
        array([5.8, 8.4])

        However, it does also work with datetime, due to their numerical basis.

        >>> xp = np.array([0, 10, 100])
        >>> a = ['2013-04-01', '2014-06-06', '2020-12-17']
        >>> fp = np.array(a, dtype=np.datetime64)
        >>> x = np.array([6, 17])
        >>> helper.interp(x, xp, fp)
        array(['2013-12-15', '2014-12-08'], dtype='datetime64[D]')
    """
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    # TODO: add checks and exceptions
    i = np.searchsorted(xp, x)
    return np.asarray(
        (x - xp[i - 1]) / (xp[i] - xp[i - 1]) * (fp[i] - fp[i - 1]) + fp[i - 1]
    )
