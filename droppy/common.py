import argparse

import numpy as np

L = 'left'
R = 'right'
T = 'top'
B = 'bottom'


def positive_int(value):
    '''
    Type checking for positive integers passed to the command-line parser

    :param value: Input that is to be type-checked (scalar)
    :return: Input cast to an int
    :raises ArgumentTypeError: If the input is less than 0 or cannot be cast
                               to an integer
    '''
    if int(value) <= 0:
        raise argparse.ArgumentTypeError(f'{value} is an invalid positive int'
                                         'value')
    return int(value)


def positive_float(value):
    '''
    Type checking for positive floats passed to the command-line parser

    :param value: Input that is to be type-checked (scalar)
    :return: Input cast to a float
    :raises ArgumentTypeError: If the input is less than 0 or cannot be cast
                               to a float
    '''
    if float(value) <= 0:
        raise argparse.ArgumentTypeError(f'{value} is an invalid positive'
                                         'float value')
    return float(value)

def positive_int_or_rel(value):
    '''
    Type checking for thresholds passed to the command-line parser that must
    either be positive integer (pixel) values or negative floats between 0 and
    -1 (relative) values.

    :param value: Input that is to be type-checked (scalar)
    :return: Input cast to an int
    :raises ArgumentTypeError: If the input is less than -1 or cannot be cast
                               to a float
    '''

    if float(value) < -1:
        raise argparse.ArgumentTypeError(f'{value} is not a relative negative value'
                                         ' or positive int')
    if int(value) > 0:
        return int(value)
    else:
        return float(value)


def calculate_angle(v1, v2):
    '''
    Compute the angle between two vectors of equal length

    :param v1: numpy array
    :param v2: numpy array
    :return: angle between the two vectors v1 and v2 in degrees
    '''
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.dot(v1_u, v2_u)) * 360 / 2 / np.pi


def baseF(x, a):
    '''
    Compute the baseline y-pixel value at a specified ``x`` position, with
    best fit parameters ``a``.

    >>> baseF(0, [10, 0])
    10

    >>> baseF(10, [10, 0])
    10

    >>> baseF(10, [10, 1])
    20

    >>> baseF(10, [10, 1, 1])
    120

    :param x: Scalar x-pixel location being probed
    :param a: Numpy array with best-fit polynomial coefficients to the
              baseline
    :return: y-pixel value of the baseline with these parameters at ``x``
    '''
    return np.dot(a, np.power(x, range(len(a))))
