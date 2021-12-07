"""
Module containing the helper functions for fitting the duper to data.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import generator
from .generator.base import Generator


def choose_generator(data: NDArray, category_threshold: float) -> Generator:
    """Chooses and returns the best generator to replicate the provided data.

    Parameters
    ---------
    data: ArrayLike
        training dataset with realistic data.
    category_threshold: float
        Fraction of unique values until which category generator is perferred,
        should be in [0,1].

    Returns
    ---------
    Generator
        the chosen generator to replicate the provided data
    """
    if len(data) == 0:
        return generator.Constant(value=np.nan, na_rate=1.0)

    if all(pd.isna(data)):
        return generator.Constant(value=pd.unique(data), na_rate=1.0)

    value_counts = pd.value_counts(data, dropna=False)
    unique_values = value_counts.index[~value_counts.index.isna()]

    if len(unique_values) == 1:
        na_rate = value_counts.get(np.nan, 0) / len(data)
        return generator.Constant(value=unique_values[0], na_rate=na_rate)

    if (
        data.dtype == np.bool_
        or len(unique_values) / len(data) < category_threshold
    ):
        return generator.Category(data=data)

    if data.dtype == np.float_:
        return generator.Float(data=data)

    if data.dtype == np.int_:
        return generator.Integer(data=data)

    if np.issubdtype(data.dtype, np.datetime64):
        return generator.Datetime(data=data)

    if data.dtype == np.str_ or data.dtype == np.object_:
        if len(set(map(len, unique_values))) == 1:
            return generator.Regex(data=data)

    return generator.Category(data=data)
