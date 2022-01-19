"""
Module containing the helper functions for fitting the duper to data.
"""
from __future__ import annotations

from typing import Type

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import generator
from .generator.base import Generator


def find_best_generator(
    data: NDArray, category_threshold: float
) -> Type[Generator]:
    """Tries to find the best generator class to replicate the provided data.

    Args:
        data (NDArray): training dataset with realistic data.
        category_threshold (float): fraction of unique values
                until which category duper is perferred, should be in [0,1].
                Defaults to 0.05.

    Returns:
        Type[Generator]: the best generator class to replicate the provided data
    """
    data = Generator.validate(data=data)

    unique_values = pd.unique(data)
    unique_values = unique_values[~pd.isna(unique_values)]

    if len(unique_values) <= 1:
        return generator.Constant

    if (
        data.dtype == np.bool_
        or len(unique_values) / len(data) < category_threshold
    ):
        return generator.Category

    if np.issubdtype(data.dtype, np.float_) or np.issubdtype(
        data.dtype, np.int_
    ):
        return generator.Numeric

    if np.issubdtype(data.dtype, np.datetime64):
        return generator.Datetime

    if data.dtype == np.str_ or data.dtype == np.object_:
        if len(set(map(len, unique_values))) == 1:
            return generator.Regex

    return generator.Category
