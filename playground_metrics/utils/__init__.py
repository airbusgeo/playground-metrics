def to_list(arr):
    """Transform an numpy array into a list with homogeneous element type."""
    return arr.astype(type(arr.ravel()[0])).tolist()


def to_builtin(arr):
    """Transform an numpy array into a numpy array with homogeneous element type."""
    return arr.astype(type(arr.ravel()[0]))
