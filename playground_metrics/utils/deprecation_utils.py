import warnings
from functools import wraps


def deprecated(reason):
    """Deprecate a function by adding a ``DeprecationWarning`` and a **Warning** block in the docstring."""
    def deprecated_decorator(func):
        @wraps(func)
        def deprecated_function(*args, **kwargs):
            warnings.warn('The function {0} is deprecated and may not work or '
                          'disappear in the future'.format(func.__name__), DeprecationWarning)
            return func(*args, **kwargs)

        former_docstring = deprecated_function.__doc__ or ''

        deprecated_function.__doc__ = \
            """
            .. warning::
                The function ``{0}`` is deprecated and may not work anymore or disappear in the future.

                Reason for deprecation: *{1}*

            """.format(func.__name__, reason) + former_docstring
        return deprecated_function
    return deprecated_decorator
