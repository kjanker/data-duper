# data-duper

data-duper is a tool to replicate the structure of private or protected data for testing.

[![PyPI version](https://img.shields.io/pypi/v/data-duper.svg?style=flat-square&label=version)](ttps://pypi.org/project/data-duper)
[![GitHub license](https://img.shields.io/github/license/kjanker/data-duper.svg)](https://github.com/kjanker/data-duper/blob/main/LICENSE)
[![CI/CD](https://github.com/kjanker/data-duper/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kjanker/data-duper/actions/workflows/ci.yml)
[![Python: ≥3.8](https://img.shields.io/badge/%20Python-≥3.8-%23FFD43B?style=flat&labelColor=4B8BBE&logo=python&logoColor=FFD43B)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/Code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20Imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## What does it solve?

When testing the data handling of software, it is best to use data as similar to the real data as possible - without revealing sensitive information to the test environment. This is where data-duper comes into play. It allows you to create an authentic replicate of your private or protected data.

## How to get it?

The source code ist currently hosted on GitHub at: https://github.com/kjanker/data-duper.

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/data-duper).

## What does it do?

data-duper works like a learning model. You train the duper on your real data and, afterwards, generate a new data set of arbitrary size. The new data set - or dupe - has the same structure as the real data, i.e., columns, dtypes, as well as string composition and distribution of numerical values. Occurences of NA values are ignored by default, but can optionally be included as well.

### Methods
- numerical values (float, int, datetime) are drawn from an interpolated empirical distribution
- identifier strings of fixed length and structure are replicated with regular expressions
- features with only few values (category, bool) are redrawn according to their occurrence

### Limitations
- value distributions are replicated as draw propability. Thus, for small dupe sets the realized distribution may differ slightly
- correlations between columns are not replicated (this ensures real data is better obscured)
- descriptive strings like notes, names, etc are not obscured but reshuffled

## How can I use it?

You simply initialize a new `Duper` instance, fit it on your real data `df_real`, and make a data dupe `df_dupe` of desired size `n`.

```python
from duper import Duper

duper = Duper()
duper.fit(df=df_real)
df_dupe = duper.make(size=10000)
```

## Open issues
- include optional correlations between selected rows
- improve algorithm of regex duper

## Get in touch

Don't hesitate to contact me if you like the idea and want to get in touch.