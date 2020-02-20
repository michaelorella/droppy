import numpy as np

from droppy.imageanalysis import crop_points
from droppy.common import L, R, T, B, baseF

def fit_line(points, order=1):
    '''
    Returns the set of coefficients of the form y = Σ a[i]*x**i for i = 0 ..
    order using np.linalg
    '''
    if len(points.shape) != 2:
        raise(IndexError('There are not enough points to fit a line, '
                         'please check this video'))

    X = np.ones((points.shape[0], order + 1))
    x = points[:, 0]
    for col in range(order + 1):
        X[:, col] = np.power(x, col)

    y = points[:, 1]
    a = np.linalg.lstsq(X, y, rcond=None)

    try:
        assert(len(a[0]) == order + 1)
    except AssertionError as e:
        print('There is something wrong with the baseline fitting procedure, '
              'the program cannot continue execution')
        raise(e)

    return a


def generate_droplet_width(crop, bounds=None, f=None):
    # Look for the greatest distance between points on the baseline
    # by calculating points that are in the circle within the linear
    # threshold

    if bounds is not None:
        just_inside = crop_points(crop, bounds, f=f)
    else:
        just_inside = crop

    limits = {L: np.amin(just_inside[:, 0]),
              R: np.amax(just_inside[:, 0])}
    return limits

def generate_vectors(linear_points, limits, ε, a, tolerance=None):

    if tolerance is None:
        tolerance = 8
    # Define baseline vector - slope will just be approximated from
    # FD at each side (i.e. limit[0] and limit[1])

    bv = {side: ([1, (baseF(pos + ε/2, a) - baseF(pos - ε/2, a)) / ε]) for
          side, pos in limits.items()}

    # Initialize paramater dictionaries
    b = {}
    m = {}

    # Initialize vector dictionary
    v = {}

    # Initialize vertical success dictionary
    vertical = {L: False, R: False}

    for side in [L, R]:
        # Get the values for this side of the drop
        fits, residual, rank, sing_vals = fit_line(linear_points[side])

        # Extract the parameters
        b[side], m[side] = fits

        # Do all the checks I can think of to make sure that fit succeeded
        if (rank != 1
                and np.prod(sing_vals) > linear_points[side].shape[0]
                and residual < tolerance):
            # Check to make sure the line isn't vertical
            v[side] = np.array([1, m[side]])
        else:
            # Okay, we've got a verticalish line, so swap x <--> y
            # and fit to c' = A' * θ'
            linear_points_prime = linear_points[side][:, ::-1]

            b[side], m[side] = fit_line(linear_points_prime)[0]

            v[side] = np.array([m[side], 1])

            vertical[side] = True

    # Reorient vectors to compute physically correct angles
    if v[L][1] > 0:
        v[L] = -v[L]
    if v[R][1] < 0:
        v[R] = -v[R]

    return v, b, m, bv, vertical
