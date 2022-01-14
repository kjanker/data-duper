import numpy as np
import pandas as pd
import pytest

from duper import generator
from duper.generator.base import Generator

# Generator


@pytest.mark.parametrize(
    "test",
    [
        {
            "data": np.datetime64("2021-12-21")
            + np.random.randint(100, size=10),
            "dtype": np.dtype("<M8[D]"),
            "nan": np.isnat,
        },
        {
            "data": np.random.randint(100, size=10),
            "dtype": np.dtype("int"),
            "nan": np.isnan,
        },
        {
            "data": np.random.uniform(size=10),
            "dtype": np.dtype("float"),
            "nan": np.isnan,
        },
    ],
)
def test_Generator_dtype(test):
    duper = Generator.from_data(data=test["data"])
    assert duper.dtype == test["dtype"]
    assert test["nan"](duper.nan)


# Constant generator


def test_ConstantGenerator_size():
    value = 3
    na_rate = 0.0
    duper = generator.Constant(value=value, na_rate=na_rate)
    duped_values = duper.make(size=100)
    assert len(duped_values) == 100


@pytest.mark.parametrize(
    "value", [0, 1, 0.4, "test", np.datetime64("2012-04-18")]
)
def test_ConstantGenerator_no_na(value):
    na_rate = 0.0
    duper = generator.Constant(value=value, na_rate=na_rate)
    duped_values = duper.make(size=100)
    assert all(duped_values == value)


def test_ConstantGenerator_few_na():
    value = 3
    na_rate = 0.1
    duper = generator.Constant(value=value, na_rate=na_rate)
    duped_values = duper.make(size=100)
    assert any(duped_values == value)


def test_ConstantGenerator_all_na():
    value = 3
    na_rate = 1.0
    duper = generator.Constant(value=value, na_rate=na_rate)
    duped_values = duper.make(size=100)
    assert any(duped_values == value)


def test_ConstantGenerator_from_data():
    data = np.array([1, 1, 1, np.nan])
    duper = generator.Constant.from_data(data)
    assert duper.dtype == data.dtype
    assert duper.na_rate == 0.25


def test_ConstantGenerator_fail_infer():
    data = np.array([1, 1, 2, np.nan])
    with pytest.raises(ValueError):
        generator.Constant.from_data(data)


# Numeric generator


def test_NumericGenerator_int():
    data = [2, 2, 2, 4, 6, 8]
    gen = generator.Numeric.from_data(data=data)

    exp_vals = np.array([2, 2, 4, 6, 8], dtype=np.int_)
    exp_bins = np.array([0.0, 0.4, 0.6, 0.8, 1.0], dtype=np.float_)

    assert gen.gcd == 2
    assert gen.dtype == np.int_
    assert gen.na_rate == 0.0
    assert all(np.equal(gen.vals, exp_vals))
    assert all(np.isclose(gen.bins, exp_bins))


def test_NumericGenerator_float():
    data = [0.2, 0.2, 0.2, 0.4, 0.6, 0.8, np.nan]
    gen = generator.Numeric.from_data(data=data)

    exp_vals = np.array([0.2, 0.2, 0.4, 0.6, 0.8], dtype=np.float_)
    exp_bins = np.array([0.0, 0.4, 0.6, 0.8, 1.0], dtype=np.float_)

    assert gen.gcd == 0.2
    assert gen.dtype == np.float_
    assert np.isclose(gen.na_rate, 1 / 7)
    assert all(np.isclose(gen.vals, exp_vals))
    assert all(np.isclose(gen.bins, exp_bins))


def test_NumericGenerator_errors():

    with pytest.raises(ValueError):
        generator.Numeric(vals=3)

    with pytest.raises(ValueError):
        generator.Numeric(vals=[2])

    with pytest.raises(ValueError):
        generator.Numeric(vals=[2, 4], bins=[0, 2, 5])

    with pytest.raises(ValueError):
        generator.Numeric(vals=[2, 4], na_rate=1.5)

    with pytest.raises(TypeError):
        generator.Numeric.from_data(
            data=np.array(
                [np.datetime64("1"), np.datetime64("1")], dtype=np.datetime64
            )
        )

    with pytest.raises(TypeError):
        generator.Numeric.from_data(data=np.array(["1", "1"], dtype=np.object_))

    with pytest.raises(ValueError):
        generator.Numeric.from_data(data=np.array([], dtype=np.int_))


# Datetime generator


@pytest.fixture(
    params=[
        {"start": "2012", "freq": "Y"},
        {"start": "2012-12", "freq": "M"},
        {"start": "2012-12-21", "freq": "D"},
        {"start": "2012-12-21T04", "freq": "h"},
        {"start": "2012-12-21T04:30", "freq": "m"},
        {"start": "2012-12-21T04:30:10", "freq": "s"},
        {"start": "2012-12-21T04:30:10.123", "freq": "ms"},
        {"start": "2012-12-21T04:30:10.123456", "freq": "ns"},
    ]
)
def datetime_data(request):
    return pd.Series(
        np.datetime64(request.param["start"]) + np.random.randint(100, size=10),
        name=request.param["freq"],
    )


def test_DatetimeGenerator_freq(datetime_data):
    duper = generator.Datetime.from_data(data=datetime_data)
    assert duper.freq == datetime_data.name


# Regex generator


@pytest.mark.parametrize(
    "regex_data",
    [
        np.array(
            list(map("-".join, zip(list("abcdefghij"), list("0123456789"))))
            + [np.nan],  # type: ignore
            dtype=object,
        ),
        pd.Series(
            list(map("-".join, zip(list("abcdefghij"), list("0123456789"))))
            + [np.nan]  # type: ignore
        ),
    ],
)
def test_RegexGenerator_regex(regex_data):
    duper = generator.Regex.from_data(data=regex_data)
    assert duper.regex == r"[a-j][\-][0-9]"
