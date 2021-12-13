"""
A generator for string data that follow a fixed structure like IDs.
"""
import itertools
import re

import numpy as np
import pandas as pd
import rstr
from numpy.typing import NDArray

from .base import Generator


class Regex(Generator):
    """Generator class recommended for strings with a repeating structure.

    It creates values from a regular expression derived from the data.

    """

    def __init__(self, data: NDArray[np.str_]) -> None:
        super().__init__(data=data)
        data_clean = data[~pd.isna(data)]
        self.na_rate = 1 - len(data_clean) / len(data)
        self.regex = self._beautify_regex(self._train_regex(data_clean))

    @classmethod
    def from_data(cls, data: NDArray):
        return cls(data=data)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with '{self.regex}'>"

    def _make(self, size: int) -> NDArray:
        return np.array([rstr.xeger(self.regex) for _ in range(size)])

    @staticmethod
    def _train_regex(data: NDArray[np.str_]) -> str:
        """Simple algorithm to derive a regular expression from a set of
        strings. It loops the strings character by character, takes the n-th
        characters of each string and builds a regular expression, allowing
        only those characters at this position.

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
