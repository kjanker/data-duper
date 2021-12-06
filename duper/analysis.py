"""
analysis.py
====================================
Module containing the helper functions for fitting the duper to data.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .methods import (
    BaseDuper,
    CategoryDuper,
    ConstantDuper,
    DatetimeDuper,
    FloatDuper,
    IntDuper,
    RegExDuper,
)


def choose_method(data: NDArray, category_threshold: float) -> BaseDuper:
    """Chooses and returns the best method to replicate the provided data.

    Parameters
    ---------
    data: ArrayLike
        training dataset with realistic data.
    category_threshold: float
        Fraction of unique values until which category duper is perferred,
        should be in [0,1].

    Returns
    ---------
    BaseDuper
        the chosen BaseDuper to replicate the provided data
    """
    if len(data) == 0:
        return ConstantDuper(value=np.nan, na_rate=1.0)

    if all(pd.isna(data)):
        return ConstantDuper(value=pd.unique(data), na_rate=1.0)

    value_counts = pd.value_counts(data, dropna=False)
    unique_values = value_counts.index[~value_counts.index.isna()]

    if len(unique_values) == 1:
        na_rate = value_counts.get(np.nan, 0) / len(data)
        return ConstantDuper(value=unique_values[0], na_rate=na_rate)

    if (
        data.dtype == np.bool_
        or len(unique_values) / len(data) < category_threshold
    ):
        return CategoryDuper(data=data)

    if data.dtype == np.float_:
        return FloatDuper(data=data)

    if data.dtype == np.int_:
        return IntDuper(data=data)

    if np.issubdtype(data.dtype, np.datetime64):
        return DatetimeDuper(data=data)

    if data.dtype == np.str_ or data.dtype == np.object_:
        if len(set(map(len, unique_values))) == 1:
            return RegExDuper(data=data)

    return CategoryDuper(data=data)
