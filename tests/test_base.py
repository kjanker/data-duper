import numpy as np
import pandas as pd
import pytest

from duper import Duper, methods


@pytest.fixture
def df_train():
    return pd.DataFrame(
        data=dict(
            const=["baz"] * 20,
            datetime=np.datetime64("2012-12-21") + np.array(range(0, 100, 5)),
            integer=np.array(range(20)) * np.array(range(20)),
            float=1 / np.array(range(1, 21)),
            string=["foo"] * 5 + ["bar"] * 10 + ["foobar"] * 5,
            regex=list(
                map("-".join, zip(list("abcdefghij"), list("0123456789")))
            )
            * 2,
        )
    )


def test_duper(df_train, size=100):
    duper = Duper()
    duper.fit(df=df_train)

    assert all(duper.columns == df_train.columns)
    assert duper.dtypes == df_train.dtypes.to_dict()
    assert list(map(type, duper.methods.values())) == [
        methods.ConstantDuper,
        methods.DatetimeDuper,
        methods.IntDuper,
        methods.FloatDuper,
        methods.CategoryDuper,
        methods.RegExDuper,
    ]

    df_dupe = duper.make(size=size)

    assert type(df_dupe) == pd.DataFrame
    assert len(df_dupe) == size
    assert all(df_dupe.columns == df_train.columns)
    assert all(df_dupe.dtypes == df_train.dtypes)


def test_NaT_column():
    df = pd.DataFrame(
        data=np.full(fill_value=np.datetime64("2012-12-12"), shape=100),
        columns=["nat"],
    )
    df["nat"] = np.datetime64("NaT")

    duper = Duper()
    duper.fit(df)
    duper.methods["nat"].value
    assert np.isnat(duper.methods["nat"].value)
    assert all(np.isnat(duper.make(size=10)))
