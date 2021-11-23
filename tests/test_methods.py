import numpy as np
import pandas as pd
import pytest

from duper.methods import BaseDuper, ConstantDuper, DatetimeDuper, RegExDuper

# BaseDuper


@pytest.mark.parametrize(
    "test",
    [
        dict(
            data=np.datetime64("2021-12-21") + np.random.randint(100, size=10),
            dtype=np.dtype("<M8[D]"),
            nan=np.isnat,
        ),
        dict(
            data=np.random.randint(100, size=10),
            dtype=np.dtype("int64"),
            nan=np.isnan,
        ),
        dict(
            data=np.random.uniform(size=10),
            dtype=np.dtype("float64"),
            nan=np.isnan,
        ),
    ],
)
def test_BaseDuper_dtype(test):
    duper = BaseDuper(data=test["data"])
    assert duper.dtype == test["dtype"]
    assert test["nan"](duper.nan)


# ConstantDuper


def test_ConstantDuper_size():
    value = 3
    na_rate = 0.0
    duper = ConstantDuper(value=value, na_rate=na_rate)
    duped_values = duper.make(n=100)
    assert len(duped_values) == 100


@pytest.mark.parametrize(
    "value", [0, 1, 0.4, "test", np.datetime64("2012-04-18")]
)
def test_ConstantDuper_no_na(value):
    na_rate = 0.0
    duper = ConstantDuper(value=value, na_rate=na_rate)
    duped_values = duper.make(n=100)
    assert all(duped_values == value)


def test_ConstantDuper_few_na():
    value = 3
    na_rate = 0.1
    duper = ConstantDuper(value=value, na_rate=na_rate)
    duped_values = duper.make(n=100)
    assert any(duped_values == value)


def test_ConstantDuper_all_na():
    value = 3
    na_rate = 1.0
    duper = ConstantDuper(value=value, na_rate=na_rate)
    duped_values = duper.make(n=100)
    assert any(duped_values == value)


# DatetimeDuper


@pytest.fixture(
    params=[
        dict(start="2012", freq="Y"),
        dict(start="2012-12", freq="M"),
        dict(start="2012-12-21", freq="D"),
        dict(start="2012-12-21T04", freq="h"),
        dict(start="2012-12-21T04:30", freq="m"),
        dict(start="2012-12-21T04:30:10", freq="s"),
        dict(start="2012-12-21T04:30:10.123", freq="ms"),
        dict(start="2012-12-21T04:30:10.123456", freq="ns"),
    ]
)
def datetime_data(request):
    return pd.Series(
        np.datetime64(request.param["start"])
        + np.random.randint(100, size=10),
        name=request.param["freq"],
    )


def test_DatetimeDuper_freq(datetime_data):
    duper = DatetimeDuper(data=datetime_data)
    assert duper.freq == datetime_data.name


# RegExDuper


@pytest.mark.parametrize(
    "regex_data",
    [
        np.array(
            pd.Series(
                list(
                    map("-".join, zip(list("abcdefghij"), list("0123456789")))
                )
                + [np.nan]
            )
        ),
        pd.Series(
            list(map("-".join, zip(list("abcdefghij"), list("0123456789"))))
            + [np.nan]
        ),
    ],
)
def test_RegExDuper_regex(regex_data):
    duper = RegExDuper(data=regex_data)
    assert duper.regex == r"[a-j][\-][0-9]"
