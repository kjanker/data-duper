User Guide
======================================

data-duper is a tool to replicate the structure of private or protected data for testing.

Installation
--------------------------------------

The source code is currently hosted on GitHub at: https://github.com/kjanker/data-duper.

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/data-duper).

.. code:: bash

    pip install data-duper

How to use the Duper
--------------------------------------

data-duper follows the scikit learn model-paradigm:

1. You create a new :class:`Duper <duper.Duper>` object.
2. :meth:`fit <duper.base.Duper.fit>` it to a pandas ``DataFrame``.
3. (Optionally) inspect or customize the fit.
4. :meth:`make <duper.base.Duper.make>` a new synthetic ``DataFrame`` of desired shape.

In the example bellow, we load public energy data (`source <https://data.open-power-system-data.org>`_)
as pandas DataFrame and create a synthetic data frame with data-duper. The duped data holds the same
columns and similar values as the original data.

.. code:: python

    import pandas as pd
    from duper import Duper
    
    # load data from https://data.open-power-system-data.org/renewable_power_plants/
    data_file = "https://data.open-power-system-data.org/renewable_power_plants/2020-08-25/renewable_power_plants_SE.csv"
    df_real = pd.read_csv(data_file, parse_dates=["commissioning_date"])

    # replicate data with duper
    duper = Duper()
    duper.fit(df_real)
    df_dupe = duper.make(size=10000)
    print(df_dupe.head())


Fitting to the data
--------------------------------------

When fitting the :class:`Duper <duper.Duper>` to a data frame, we use :meth:`find_best_generator <duper.analysis.find_best_generator>`
to derive and configure the best method to dupe each column. This approach considers the data type
and values of each column, and creates a :class:`Generator <duper.generator.base.Generator>` to dupe
that column's data.

Note that we currently do not account for relations between columns, but consider each column separately.

NA values in the data are aggregated to an :meth:`na_rate <duper.generator.base.Generator.na_rate>`

* *Constant* represents columns that only hold a single value of any type. See :class:`Constant <duper.generator.Constant>` generator.
* *Category* is designed for data with few different values of any type. This is also the fallback generator if no other fits. See :class:`Category <duper.generator.Category>` generator.
* *Numerical* data is fitted to an empirical distribution function. The original data type and granularity of the values is taken into account. However, we currently do not capture continuous index-like nummers. Id-like numbers of a fixed length can be cast as strings to be duped via regex. See :class:`Numeric <duper.generator.Numeric>` generator.
* *DataTime* data is currently fitted similar to numerical data. Hence, it works great for unordered dates and times but does not work not capture continuous index-like timestamps of a certain frequency. See :class:`Datetime <duper.generator.Datetime>` generator.
* *Regex* helps to replicate ids, serial numbers, and other special strings using a regular expression. See :class:`Regex <duper.generator.Regex>` generator.

Inspect and edit generators
--------------------------------------

You might want to inspect your :class:`Duper <duper.Duper>` after fitting to a data set. This is done
by use of the the attributes ``columns``, ``dtypes``, and ``generators``. Alternatively, you can get
and set a columns generator also directly with the column name in square brackets. This can also be used
to add new columns manually.

.. code:: python

    import pandas as pd
    from duper import Duper
    from duper.generators import Constant
    
    # load data from https://data.open-power-system-data.org/renewable_power_plants/
    data_file = "https://data.open-power-system-data.org/renewable_power_plants/2020-08-25/renewable_power_plants_SE.csv"
    df_real = pd.read_csv(data_file, parse_dates=["commissioning_date"])

    # replicate data with duper
    duper = Duper()
    duper.fit(df_real)

    # insprect duper
    print(duper.columns)
    print(duper.dtypes)
    print(duper.generators)

    # inspect specific column generators
    duper["commissioning_date"]

    # set/overwrite a specific generators
    duper["manufacturer"] = Constant(value="Umbrella Corp.")
    duper["my_new_column"] = Constant(value=True)

Make synthetic dataset
--------------------------------------

A fitted :class:`Duper <duper.base.Duper>` can be used to :meth:`make <duper.base.Duper.make>` a random
new data frame that replicates the original one. Since the data is generated randomly, the number of rows (``size``)
can be set freely.

Voila, you have created a new data frame that replicates the original structure.

The Duper class
--------------------------------------

.. autoclass:: duper.base.Duper
       
   .. rubric:: Methods

   .. autosummary::
   
      ~Duper.__init__
      ~Duper.fit
      ~Duper.make


   .. rubric:: Attributes

   .. autosummary::
   
      ~Duper.columns
      ~Duper.dtypes
      ~Duper.generators

Methods
~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: duper.base

.. automethod:: Duper.__init__
.. automethod:: Duper.fit
.. automethod:: Duper.make


Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: duper.base

.. autoattribute:: Duper.columns
.. autoattribute:: Duper.dtypes
.. autoattribute:: Duper.generators