import numpy as np

import scipy.optimize as opt

def dist(param, points):
    '''
    Calculate the total distance from the calculated circle to the points

    :param param:  list of 3 elements with the center of a circle and its
                   radius
    :param points: list of (x, y) points that should be lying on the circle
    :return: the sum of squared errrors from the points on the circle with
             the provided parameters
    '''
    *z, r = param
    ar = [(np.linalg.norm(np.array(z) - np.array(point)) - r) ** 2
          for point in points]
    return np.sum(ar)


def fit_circle(points, width=None, start=False):
    '''
    Compute the best-fit circle to ``points`` by minimizing ``dist`` from
    changing values of the centerpoint and radius

    :param points: list of (x,y) points lying on the droplet to fit
    :param width: estimated width of the droplet from the crop boundaries
    :param start: boolean flag to determine how many parameters to fit (just
                  radius if True, otherwise radius and centerpoint)
    :return: result structure from scipy.opt.minimize
    '''
    if width is None:
        width = np.max(points[:, 0]) - np.min(points[:, 1])

    if start:
        # Try to fit a circle to the points that we have extracted,
        # only varying the radius about the center of all the points
        z = np.mean(points, axis=0)
        res = opt.minimize(lambda x: dist([*z, x], points),
                           width / 2)

        # Get the results
        res['x'] = np.array([*z, res['x'][0]])
    else:
        # Fit this new set of points, using the full set of parameters
        res = opt.minimize(lambda x: dist(x, points),
                           np.concatenate((np.mean(points, axis=0),
                                           [width / 4])))

    return res


def generate_circle_vectors(intersection):
    '''
    Using the intersection point with the baseline, compute the vector that
    points tangent to the circle

    :param intersection: (x,y) point on the circle that crosses the baseline
    :return: baseline vector and vector tangent to best-fit circle
    '''
    x_t, y_t = intersection
    # For contact angle, want interior angle, so look at vector in
    # negative x direction (this is our baseline)
    v1 = np.array([-1, 0])

    # Now get line tangent to circle at x_t, y_t
    if y_t != 0:
        slope = - x_t / y_t
        v2 = np.array([1, slope])
        v2 = v2 / np.linalg.norm(v2)
        if y_t < 0:
            # We want the interior angle, so when the line is
            # above the origin (into more negative y), look left
            v2 = -v2
    else:
        v2 = np.array([0, 1])

    return v1, v2


def find_intersection(baseline_coeffs, circ_params):
    '''
    Compute the intersection points between the best fit circle and best-fit
    baseline.

    For this we rely on several coordinate transformations, first a
    translation to the centerpoint of the circle and then a rotation to give
    the baseline zero-slope.

    :param baseline_coeffs: Numpy array of coefficients to the baseline
                            polynomial
    :param circ_params: centerpoint and radius of best-fit circle
    :return: (x,y) point of intersection between these two shapes
    '''
    *z, r = circ_params
    b, m = baseline_coeffs[0:2]
    # Now we need to actually get the points of intersection
    # and the angles from these fitted curves. Rather than brute force
    # numerical solution, use combinations of coordinate translations and
    # rotations to arrive at a horizontal line passing through a circle.
    # First step will be to translate the origin to the center-point
    # of our fitted circle
    # x = x - z[0], y = y - z[1]
    # Circle : x**2 + y**2 = r**2
    # Line : y = m * x + (m * z[0] + b - z[1])
    # Now we need to rotate clockwise about the origin by an angle q,
    # s.t. tan(q) = m
    # Our transformation is defined by the typical rotation matrix
    #   [x;y] = [ [ cos(q) , sin(q) ] ;
    #             [-sin(q) , cos(q) ] ] * [ x ; y ]
    # Circle : x**2 + y**2 = r**2
    # Line : y = (m*z[0] + b[0] - z[1])/sqrt(1 + m**2)
    # (no dependence on x - as expected)

    # With this simplified scenario, we can easily identify the points
    # (x,y) where the line y = B
    # intersects the circle x**2 + y**2 = r**2
    # In our transformed coordinates, only keeping the positive root,
    # this is:

    B = (m * z[0] + b - z[1]) / np.sqrt(1 + m**2)

    if B > r:
        raise ValueError("The circle and baseline do not appear to intersect")
    x_t = np.sqrt(r ** 2 - B ** 2)
    y_t = B

    # TODO:// replace the fixed linear baseline with linear
    # approximations near the intersection points

    return x_t, y_t
