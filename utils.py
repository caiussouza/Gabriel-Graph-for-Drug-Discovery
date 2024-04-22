def sign(x):
    """Sign function

    Args:
        x (int or float): Argument

    Returns:
        int: 1 if x is positive or zero, -1 if x is negative.
    """
    return 2 * (x >= 0) - 1
