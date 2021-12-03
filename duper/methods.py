"""
methods.py
====================================
Module containing the replication methods for each type of data.
"""
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


class BaseDuper:
    """Abstract class of the value generators."""

    def __init__(self, data: ArrayLike):
        self.dtype = data.dtype
        if self.dtype.name.startswith("datetime"):
            self.nan = np.datetime64("NaT")
        else:
            self.nan = np.nan

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def _make(n: int):
        pass

    def make(self, n: int, with_na: bool = False) -> NDArray:
        data = self._make(n=n).astype(self.dtype)
        if with_na:
            isna = np.random.uniform(size=n) < self.na_rate
            data[isna] = self.nan
        return data


class ConstantDuper(BaseDuper):
    """Simplest duper method. Replicates data with a constant value."""

    def __init__(self, value, na_rate: float = 0.0):
        super().__init__(data=np.array(value))
        self.value = value
        self.na_rate = na_rate

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with value '{self.value}'"

    def _make(self, n: int) -> ArrayLike:
        return np.full(shape=n, fill_value=self.value)


class CategoryDuper(BaseDuper):
    """Recommended for string data of few different values.
    Draws values based on their occurence in the data.

    """

    def __init__(self, data: ArrayLike):
        super().__init__(data=data)
        self._choices = pd.value_counts(data, normalize=True, dropna=False)

    def __str__(self) -> str:
        catagories_str = self.categories(with_na=True).to_string(
            header=False,
            index=True,
            length=False,
            dtype=False,
            name=False,
            na_rep="NA",
        )
        return (
            f"{self.__class__.__name__} with "
            f"{len(self.categories(with_na=True))} values:\n"
            f"{catagories_str}"
        )

    def categories(self, with_na=True) -> pd.Series:
        if with_na:
            return self._choices
        else:
            choices = self._choices[~self._choices.index.isna()]
            return choices / choices.sum()

    def make(self, n: int, with_na=False) -> NDArray:
        return np.random.choice(
            a=self.categories(with_na=with_na).index,
            size=n,
            p=self.categories(with_na=with_na),
        )


class QuantileDuper(BaseDuper):
    """Meta duper class recommented for numerical data. Do not use directly.

    Replicates the data by drawing from the linear interpolated qunatile.

    """

    def __init__(self, data: ArrayLike):
        super().__init__(data=data)
        self.data = data[~np.isnan(data)]
        self.na_rate = 1 - len(self.data) / len(data)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} from empiric quantiles"

    def _make(self, n: int) -> NDArray:
        p = np.random.uniform(0, 1, n)
        return np.quantile(self.data, p, interpolation="linear")


class FloatDuper(QuantileDuper):
    """Duper class recommended to replicate continous float data.

    As methods, this is directly based on the meta QuantileDuper class.

    """

    pass


class IntDuper(QuantileDuper):
    """Duper class recommended to replicate integer data.

    As methods, this is based on the meta QuantileDuper class.

    """

    def _make(self, n: int) -> NDArray:
        return super()._make(n=n).round()


class DatetimeDuper(QuantileDuper):
    """Duper class recommended to replicate datetime data.

    As methods, this is based on the meta QuantileDuper class.

    """

    def __init__(self, data: ArrayLike, freq: str = None):
        super().__init__(data=data)
        if freq is None:
            self._set_auto_freq()
        else:
            self.data = self.data.astype(f"datetime64[{freq}]")
            self.freq = freq

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} from empiric quantiles, "
            f"freq={self.freq}"
        )

    def _set_auto_freq(self):
        """Derive datetime frequency dtype from data"""
        self.freq = "ns"
        for freq in ["ms", "s", "m", "h", "D", "M", "Y"]:
            if any(self.data != self.data.astype(f"datetime64[{freq}]")):
                break
            else:
                self.data = self.data.astype(f"datetime64[{freq}]")
                self.freq = freq


class RegExDuper(BaseDuper):
    """Duper class recommended to strings with a repeating defined structure.

    It creates values from a regular expression derived from the data.

    """

    def __init__(self, data: ArrayLike):
        super().__init__(data=data)
        data_clean = data[~pd.isna(data)]
        self.na_rate = 1 - len(data_clean) / len(data)
        self.regex = self._beautify_regex(self._train_regex(data_clean))

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with '{self.regex}'>"

    def _make(self, n: int) -> NDArray:
        from rstr import xeger

        return np.array([xeger(self.regex) for _ in range(n)])

    @staticmethod
    def _train_regex(data: ArrayLike) -> str:
        """Simple algorithm to derive a regular expression from a set of
        strings. It loops the strings character by character, takes the n-th
        characters of each string and builds a regular expression, allowing
        only those characters at this position.

        """
        from itertools import zip_longest

        # break words in data into list of characters
        l = map(list, data)
        # transpose character matrix: sublists hold i-th char of each value
        l = map(list, zip_longest(*l, fillvalue=""))
        # reduce list to unique values
        l = map(set, l)
        l = map(list, l)
        # sort regex to be more readable
        l = map(np.sort, l)
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
        l = [list(map(lambda c: replace_dict.get(c, c), ll)) for ll in l]
        # merge lists of characters to regex
        l = map("".join, l)
        l = map(lambda x: f"[{x}]", l)
        return "".join(l)

    @staticmethod
    def _beautify_regex(regex: str) -> str:
        from re import sub

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
        return sub(r"\-[a-zA-Z0-9]\-", "-", regex)
