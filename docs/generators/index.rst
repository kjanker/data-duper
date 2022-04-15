
Generators
======================================

.. currentmodule:: duper.generator.base
.. autoclass:: Generator

    .. rubric:: Methods

    .. autosummary::

        ~Generator.make
        ~Generator.validate

    .. rubric:: Attributes

    .. autosummary::
    
        ~Generator.DATA_DTYPES
        ~Generator.dtype
        ~Generator.na_rate
        ~Generator.nan

The following generators are implemented:

.. currentmodule:: duper.generator
.. autosummary::
    :template: class.rst
    :toctree: ../generators
    :recursive:

    Category
    Constant
    Numeric
    Datetime
    Regex