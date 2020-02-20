import numba

import numpy as np

import scipy.optimize as opt
from scipy.spatial import distance
from scipy.integrate import solve_ivp

@numba.jit(nopython=True)
def bashforth_adams(t, y, a, b):
    '''
    Rate of change of point on droplet profile with respect to the parametric
    angle ϕ.

    :param t: Parameteric angle ϕ
    :param y: (x, z) point on the droplet profile
    :param a: capillary length in units px
    :param b: radius of curvature at apex in units of px
    :return: rate of change of each x and z with respect to ϕ
    '''
    x, z = y
    t = t / 180 * np.pi
    dxdphi = b*x*np.cos(t) / (a**2 * b * x * z + 2 * x - b * np.sin(t))
    dzdphi = b*x*np.sin(t) / (a**2 * b * x * z + 2 * x - b * np.sin(t))
    return dxdphi, dzdphi

def sim_bashforth_adams(h, a=1, b=1, all_the_way=False):
    '''
    Simulates the full profile of the Bashforth-Adams droplet from the apex

    Starts at x = +-1e-5, z = 0 and integrates to z = h along the curve
    defined by the ``bashforth-adams`` function

    :param h: Height of the droplet in px
    :param a: Capillary length of the fluid
    :param b: Curvature at the apex
    :param all_the_way: Boolean to determine whether to stop at z==h or ϕ==180
    :return: List of ϕ and (x, z) coordinates where the solver executed
    '''
    height = lambda t, y, a, b: y[1] - h
    height.terminal = True
    if all_the_way:
        height.terminal = False

    sol_l = solve_ivp(bashforth_adams, (0, -180) , (1e-5, 0), args=(a, b, ),
                      method='BDF',
                      t_eval=np.linspace(0, -180, num=500), events=height)
    sol_r = solve_ivp(bashforth_adams, (0, 180) , (1e-5, 0), args=(a, b, ),
                      method='BDF',
                      t_eval=np.linspace(0, 180, num=500), events=height)

    angles = np.hstack((sol_l.t, sol_r.t[::-1])).T
    pred = np.vstack([np.hstack((sol_l.y[0],sol_r.y[0][::-1])),
                      np.hstack((sol_l.y[1],sol_r.y[1][::-1]))]).T

    return angles, pred

def fit_bashforth_adams(data, a=0.1, b=3):
    '''
    Calculates the best-fit capillary length and curvature at the apex given
    the provided data for the points on the edge of the droplet

    :param data: list of (x, y) points of the droplet edges
    :param a: initial guess of capillary length
    :param b: initial guess of curvature at the apex
    :return: solution structure from scipy.opt.minimize
    '''
    def calc_error(h, params):
        '''
        Calulate the sum-squared error between the points on the curve and
        the measured data points

        :param h: Height of the droplet in pixels
        :param params: tuple of capillary length and curvature at apex
        :return: sum-squared error between points on the curve and data
        '''
        a, b = params

        _, pred = sim_bashforth_adams(h, a=a, b=b)

        dist = distance.cdist(data, pred)
        return np.linalg.norm(np.min(dist, axis=1))

    x_0 = (a, b)
    bounds = [[0,10], [0, 100]]

    h = np.max(data[:, 1])
    optimum = opt.minimize(lambda x: calc_error(h, x), x_0,
                           method='Nelder-Mead',
                           options={'disp':False})
    return optimum
