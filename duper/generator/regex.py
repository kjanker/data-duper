"""
A generator for string data that follow a fixed structure like IDs.
"""
from __future__ import annotations

import itertools
import re
from typing import List

import numpy as np
import pandas as pd
import rstr
from numpy.typing import DTypeLike, NDArray

from .base import Generator


class Regex(Generator):
    """Generator class recommended for strings with a repeating structure.

    It creates values from a regular expression and is best used for data that
    follow a fixed structure like IDs.

    """

    DATA_DTYPES: List[DTypeLike] = [np.str_, np.object_]

    def __init__(
        self, regex: str, dtype: DTypeLike = None, na_rate: float = 0.0
    ):
        """Initializes a new generator based on the provided regular expression.
        Consider using :meth:`from_data()` to fit the expression from data.

        Args:
            regex (str): a valid regular expression
            dtype (DTypeLike, optional): by default, the dtype is derived from
                **value**. Optionally, a valid numpy dtype can be provided.
            na_rate (float, optional): rate at which NA occour in the data.
                Must be in [0,1]. Defaults to 0.0.
        """
        self.regex = regex
        self.dtype = dtype
        self.na_rate = na_rate

    @classmethod
    def from_data(cls, data: NDArray):
        """Initializes a new generator from the provided **data**. This is
        a convenience interface and the preferred method to create a new
        generator instance.

        The provided data is interpreted as string and used to train a regular
        expression that can later be used to generate new data points.

        Args:
            data (NDArray): set of data, interpreted as strings
        """
        data = cls.validate(data=data)
        vals = np.asarray(data[~pd.isna(data)]).astype(np.str_)
        na_rate = 1 - vals.size / data.size
        regex = cls._beautify_regex(cls._train_regex(vals))
        return cls(regex=regex, dtype=data.dtype, na_rate=na_rate)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with '{self.regex}'>"

    def _make(self, size: int) -> NDArray:
        """Hidden maker method. Creates an array of the given **size**
        by building strings from the generator's regular expression.

        Args:
            size (int): number of elements in returned array.

        Returns:
            NDArray: array of strings.
        """
        return np.array([rstr.xeger(self.regex) for _ in range(size)])

    @staticmethod
    def _train_regex(data: NDArray[np.str_]) -> str:
        """Simple algorithm to fit a regular expression on a set of strings.

        It loops the strings character by character, takes the n-th
        characters of each string and builds a regular expression, allowing
        only those characters at this position.

        Consider postprocessing the result with :meth:`_beautify_regex()`.

        Args:
            data (NDArray): array of strings, must not contain NA.

        Returns:
            str: raw regular expression, potentially bloated.

        """
        # break words in data into list of characters
        char_array = map(list, data)
        # transpose character matrix: sublists hold i-th char of each value
        char_array_transposed = map(
            list, itertools.zip_longest(*char_array, fillvalue="")
        )
        # reduce list to unique values and sort
        unique_chars = map(
            np.sort,
            np.array(
                list(map(list, map(set, char_array_transposed))), dtype=object
            ),
        )
        # account for special regex characters
        replace_dict = {
            ".": r"\.",
            "^": r"\^",
            "$": r"\$",
            "*": r"\*",
            "+": r"\+",
            "-": r"\-",
            "?": r"\?",
            "(": r"\(",
            ")": r"\)",
            "[": r"\[",
            "]": r"\]",
            "{": r"\{",
            "}": r"\}",
            "\\": r"\\\\",
            "|": r"\|",
            "/": r"\/",
        }
        regex = map(
            lambda x: f"[{x}]",
            map(
                "".join,
                [
                    map(lambda c: replace_dict.get(c, c), uc)
                    for uc in unique_chars
                ],
            ),
        )
        # merge lists of characters to regex
        return "".join(regex)

    @staticmethod
    def _beautify_regex(regex: str) -> str:
        """Makes regular expressions shorter and more elegant. This is used as
        postprocessing after :meth:`_train_regex()`.

        Args:
            regex (str): a bloated regular expression

        Returns:
            str: a more compact regular expression

        Examples:
            >>> Regex._beautify_regex('[ABCD][123789][xyz]')
            '[A-D][1-37-9][x-y]'
        """
        for chars in [
            range(ord("0"), ord("9") - 1),
            range(ord("a"), ord("z") - 1),
            range(ord("A"), ord("Z") - 1),
        ]:
            for i in chars:
                find_str = chr(i) + chr(i + 1) + chr(i + 2)
                replace_str = chr(i) + "-" + chr(i + 2)
                regex = regex.replace(find_str, replace_str)
                if i not in [ord("9") - 2, ord("a") - 2, ord("Z") - 2]:
                    find_str = "-" + chr(i + 2) + chr(i + 3)
                    replace_str = "-" + chr(i + 3)
                    regex = regex.replace(find_str, replace_str)
        return re.sub(r"\-[a-zA-Z0-9]\-", "-", regex)
