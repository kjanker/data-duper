# data-duper

data-duper is a tool to replicate the structure of private or protected data for testing.

## What does it solve?

When testing the data handling of software, it is best to use data as similar to the real data as possible - without revealing sensitive information to the test environment. This is where data-duper comes into play. It allows you to create an authentic replicate of your private or protected data.

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
df_dupe = duper.make(n=10000)
```

## Open issues
- include optional correlations between selected rows
- improve algorithm of regex duper

## Get in touch

Don't hesitate to contact me if you like the idea and want to get in touch.