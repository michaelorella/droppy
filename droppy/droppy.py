import os

import argparse

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

import re

import numpy as np

from skimage import io
from skimage.transform import hough_line, hough_line_peaks

from time import sleep

from droppy.common import (L, R, T, B, positive_int, positive_float,
                           calculate_angle, positive_int_or_rel)
from droppy.edgedetection import (sigma_setter, extract_edges)
from droppy.moviehandling import (extract_grayscale_frames,
                                          output_plots,
                                          output_datafile)
from droppy.imageanalysis import (get_crop, auto_crop, output_fits,
                                  crop_points, output_text)
from droppy.linearfits import (generate_droplet_width, generate_vectors,
                               fit_line)
from droppy.circularfits import (generate_circle_vectors, find_intersection,
                                 fit_circle)
from droppy.bafits import sim_bashforth_adams, fit_bashforth_adams

def analyze_frame(im, time, bounds, circ_thresh,
                  lin_thresh, σ, low, high, ε, lim, fit_type):
    '''
    Report the main findings for a single contact angle image

    Takes the provided image and fits it with the specified method ['linear',
    'circular', 'bashforth-adams'] to calculate the droplet contact angle,
    baseline width, and volume.

    Its main use lies within the DropPy main script, but it can also be used
    externally for debugging individual frames of a video file.

    :param im: 2D numpy array of a grayscale image
    :param time: float value of the movie time after burn-in
    :param bounds: edges of the box which crop the image
    :param circ_thresh: height above which the baseline does not exist
    :param lin_thresh: distance that preserves a set of linear points on the
                       droplet
    :param σ: value of the Gaussian filter used in the Canny algorithm
    :param low: value of the weak pixels used in dual-thresholding
    :param high: value of the strong pixels used in dual-thresholding
    :param ε: size of finite difference step to take in approximating baseline
              slope
    :param lim: maximum number of iterations to take during circle fitting
    :param fit_type: specified method for fitting the droplet profile
    :return: 5-tuple of (L, R) contact angles, contact area diameter,
             calculated droplet volume, fitted (x, y) points on droplet, and
             fitted (x,y) points on baseline
    '''
    coords = extract_edges(im, σ=σ, low=low, high=high)

    if bounds is None:
      bounds = auto_crop(im, σ=σ, low=low, high=high)

    crop = crop_points(coords, bounds)

    cropped_edges = np.zeros((np.max(crop[:,1])+1, np.max(crop[:,0])+1), dtype=bool)

    for pt in crop:
      cropped_edges[pt[1], pt[0]] = True

    # Get the baseline from the linear Hough transform
    accums, angles, dists = hough_line_peaks(*hough_line(cropped_edges), num_peaks=5)

    # Change parameterization from (r, θ) to (m, b) for standard form of line
    a = [dists[0]/np.sin(angles[0]), -np.cos(angles[0])/np.sin(angles[0])]

    f = {i: lambda x, y: x for i in [L, R]}
    f[B] = lambda x, y: y - (np.dot(a, np.power(x, range(len(a)))))
    f[T] = lambda x, y: y

    b = np.copy(bounds)
    b[3] = - circ_thresh
    circle = crop_points(crop, b, f=f)

    # Make sure that flat droplets (wetted) are ignored
    # (i.e. assign angle to NaN and continue)
    if circle.shape[0] < 5:
        return (np.nan, np.nan), np.nan, np.nan, np.array(((np.nan, np.nan),(np.nan, np.nan))), np.array(((np.nan, np.nan),(np.nan, np.nan)))

    # Baseline
    x = np.linspace(0, im.shape[1])
    y = np.dot(a, np.power(x, [[po]*len(x)
                               for po in range(2)]))

    baseline = np.array([x, y]).T

    if fit_type == 'linear':
        b = np.copy(bounds)
        b[3] = -(circ_thresh + lin_thresh)
        limits = generate_droplet_width(crop, b, f)

        # Get linear points
        f[T] = f[B]
        linear_points = {L: crop_points(crop,
                                        [int(limits[L]-2*lin_thresh),
                                         int(limits[L]+2*lin_thresh),
                                         -(circ_thresh+lin_thresh),
                                         -circ_thresh], f=f),
                         R: crop_points(crop,
                                        [int(limits[R]-2*lin_thresh),
                                         int(limits[R]+2*lin_thresh),
                                         -(circ_thresh+lin_thresh),
                                         -circ_thresh], f=f)}

        if linear_points[L].size == 0 or linear_points[R].size == 0:
            raise IndexError('We could not identify linear points, '
                             f'try changing lin_thresh from {lin_thresh}')

        v, b, m, bv, vertical = generate_vectors(linear_points,
                                                 limits, ε, a)

        # Calculate the angle between these two vectors defining the
        # base-line and tangent-line
        ϕ = {i: calculate_angle(bv[i], v[i]) for i in [L, R]}

        fit = {}
        # Plot lines
        for side in [L, R]:
            x = np.linspace(0, im.shape[1])
            if not vertical[side]:
                y = m[side] * x + b[side]
            else:
                y = np.linspace(0, im.shape[0])
                x = m[side] * y + b[side]
            fit[side] = np.array([x, y]).T

        baseline_width = limits[R] - limits[L]

        volume = np.NaN
        # TODO:// Add the actual volume calculation here!

    elif fit_type == 'circular' or fit_type == 'bashforth-adams':
        # Get the cropped image width
        width = bounds[1] - bounds[0]

        res = fit_circle(circle, width, start=True)
        *z, r = res['x']

        theta = np.linspace(0, 2 * np.pi, num=500)
        x = z[0] + r * np.cos(theta)
        y = z[1] + r * np.sin(theta)

        iters = 0

        # Keep retrying the fitting while the function value is
        # large, as this indicates that we probably have 2 circles
        # (e.g. there's something light in the middle of the image)
        while res['fun'] >= circle.shape[0] and iters < lim:

            # Extract and fit only those points outside
            # the previously fit circle
            points = np.array([(x, y) for x, y in circle
                              if (x - z[0]) ** 2 + (y - z[1]) ** 2
                              >= r ** 2])
            res = fit_circle(points, width)
            *z, r = res['x']
            iters += 1

        x_t, y_t = find_intersection(a, res['x'])

        v1, v2 = generate_circle_vectors([x_t, y_t])

        ϕ = {i: calculate_angle(v1, v2) for i in [L, R]}
        if fit_type == 'circular':
            baseline_width = 2 * x_t

            volume = (2/3 * np.pi * r ** 3
                      + np.pi * r ** 2 * y_t
                      - np.pi * y_t ** 3 / 3)

            # Fitted circle
            theta = np.linspace(0, 2 * np.pi, num=100)
            x = z[0] + r * np.cos(theta)
            y = z[1] + r * np.sin(theta)
            fit = np.array([x, y]).T
        else:
            # Get points within 10 pixels of the circle edge
            points = np.array([(x, y) for x, y in circle
                               if (x - z[0]) ** 2 + (y - z[1]) ** 2
                               >= (r-10) ** 2])

            points[:, 1] = - np.array([y - np.dot(a, np.power(y,
                             range(len(a)))) for y in points[:, 1]])
            center = (np.max(points[:, 0]) + np.min(points[:, 0]))/2
            points[:, 0] = points[:, 0] - center
            h = np.max(points[:, 1])
            points = np.vstack([points[:, 0],
                                h - points[:, 1]]).T

            cap_length, curv = fit_bashforth_adams(points).x
            θs, pred = sim_bashforth_adams(h, cap_length, curv)
            ϕ[L] = -np.min(θs)
            ϕ[R] = np.max(θs)

            θ = (ϕ[L] + ϕ[R])/2

            R0 = pred[np.argmax(θs),0] - pred[np.argmin(θs),0]
            baseline_width = R0

            P = 2*cap_length/ curv
            volume = np.pi * R0 * (R0 * h + R0 * P - 2 * np.sin(θ))
            x = pred[:, 0] + center
            y = np.array([np.dot(a, np.power(y, range(len(a)))) + y
                          for y in (pred[:, 1] - h)])
            fit = np.array([x, y]).T

    else:
        raise Exception('Unknown fit type! Try another.')

    # FI FITTYPE
    output_text(time, ϕ, baseline_width, volume)

    return (ϕ[L], ϕ[R]), baseline_width, volume, fit, baseline

def parse_cmdline(argv=None):
    '''
    Extract command line arguments to change program execution

    :param argv: List of strings that were passed at the command line
    :return: Namespace of arguments and their values
    '''

    parser = argparse.ArgumentParser(description='Calculate the contact '
                                     'angles '
                                     'from the provided image(s) or '
                                     'video file')
    parser.add_argument('path', help='Relative or absolute path to '
                                     'either image/video file(s) to be '
                                     'analyzed, or a directory in which to '
                                     'analyze all video/image files. If multiple '
                                     'files are provided, filenames should '
                                     'be separated by a space.',
                        default='./', nargs='*')
    parser.add_argument('-c', '--circleThreshold', type=positive_int_or_rel,
                        default=5,
                        help='Number of pixels above the baseline at which '
                             'points are considered on the droplet. To define'
                             ' the threshold relative to the image size, the'
                             ' value should be a float between -1 and 0.',
                        action='store', dest='circ_thresh')
    parser.add_argument('-l', '--linearThreshold', type=positive_int_or_rel,
                        default=10,
                        action='store', dest='lin_thresh',
                        help='The number of pixels inside the circle which '
                             'can be considered to be linear and should be '
                             'fit to obtain angles. To define'
                             ' the threshold relative to the image size, the'
                             ' value should be a float between -1 and 0.')
    parser.add_argument('-f', '--frequency', type=positive_float, default=1,
                        action='store', dest='frame_rate',
                        help='Frequency at which to analyze images from a '
                             'video')
    parser.add_argument('--sigma', type=float, default=1,
                        action='store', dest='σ',
                        help='Initial image filter used for edge detection')
    parser.add_argument('--checkFilter', action='store_true',
                        help='Set flag to check the provided filter value '
                             'or procede without any confirmation')
    parser.add_argument('-s', '--startSeconds', type=positive_float,
                        default=10,
                        action='store', dest='video_start_time',
                        help='Amount of time in which video should be burned '
                             'in before beginning analysis')
    parser.add_argument('--fitType', choices=['linear', 'circular',
                                              'bashforth-adams'],
                        default='bashforth-adams', type=str, action='store',
                        dest='fit_type',
                        help='Type of fit to perform to identify the contact '
                             'angles')
    parser.add_argument('--tolerance', type=positive_float, dest='ε',
                        action='store', default=1e-2,
                        help='Finite difference tolerance')
    parser.add_argument('--maxIters', type=positive_int, dest='lim',
                        action='store', default=10,
                        help='Maximum number of circle fitting iterations')
    parser.add_argument('--blockAtEnd', action='store_true',
                        dest='block_at_end',
                        help='Flag to keep plots at the end of the script '
                             'open for user review')
    parser.add_argument('--saveFigs', action='store', dest='savefile',
                        default=None, help='Base file path (without '
                        'extensions where saved figures should go')
    parser.add_argument('--crop', action='store_false',
                        dest='auto_crop',
                        help='Flag to disable automation of the image cropping '
                             'using Hough Transforms')
    parser.add_argument('-k', '--keyword', type=str, dest='keyword',
                        action='store', default='', help='Keyword argument '
                        'to match certain files in the directory, will be '
                        'ignored if the path is a single file')
    parser.add_argument('-r', '--redo', action='store_true', dest='redo',
                        help='Flag to recalculate results for path, whether '
                        'it has already been performed or not')
    parser.add_argument('--nproc', action='store', dest='nproc',
                        help='Number of processes to use for the parallel '
                        'computation. If nproc == 1, the frames are analyzed '
                        'in serial.', default=1)
    parser.add_argument('--savefig', action='store', dest='savefig',
                        help='Absolute or relative path to the image file'
                        'used to store the output figure.', default=None)

    args = parser.parse_args(argv)

    return args

def main(argv=None):
    '''
    Main analysis method for DropPy package

    This is the main entry point to the DropPy package for most users, and
    will serve as the most useful call signature. In addition to providing
    a functional form for accessing the DropPy software, the same features
    are provided by the ``droppy`` script at any command line interface where
    DropPy is installed. To access via the command line interface, simply use::

    $ droppy "filename" --fitType bashforth-adams

    The above call will analyze the video/image/directory using the Bashforth-
    Adams fitting method, and will output the results in a csv of the same
    name for further visualization/processing. The different options for this
    script can be found by running::

    $ droppy --help

    Options:

    -h, --help               show this help message and exit
    -c, --circleThreshold    Number of pixels above the baseline at which
                             points are considered on the droplet
    -l, --linearThreshold    The number of pixels inside the circle which can
                             be considered to be linear and should be fit to
                             obtain angles
    -f, --frequency          Frequency at which to analyze images from a video
    --sigma                  Initial image filter used for edge detection
    --checkFilter            Set flag to check the provided filter value or
                             procede without any confirmation
    -s, --startSeconds       Amount of time in which video should be burned in
                             before beginning analysis
    --fitType                {linear,circular,bashforth-adams}
                             Type of fit to perform to identify the contact
                             angles
    --tolerance              Finite difference tolerance
    --maxIters               Maximum number of circle fitting iterations
    --blockAtEnd             Flag to keep plots at the end of the script open
                             for user review
    --crop                   Flag to prevent automatic cropping of the image using
                             Hough transforms
    -k, --keyword            Keyword argument to match certain files in the
                             directory, will be ignored if the path is a
                             single file
    -r, --redo               Flag to recalculate results for path, whether it
                             has already been performed or not
    --nproc                  Number of processes to use for the parallel
                             computation. If nproc == 1, the frames are
                             analyzed in serial.

    To use this function within an interactive Python session, the function
    can be called explicitly.

    >>> main(['filename','--fitType', 'bashforth-adams'])

    This will result in exactly the same behavior as the script, but provides
    an interactive interface.

    :param argv: String command line arguments passed to the parser
    :return: Outputs a text-based table of contact angles and baseline widths
             to the command line from which the program is run. Additionally,
             several simple plots are generated including a comparison
             between the original image files and the calculated fits and the
             contact angle and baseline width over time. Finally, these data
             are exported to a csv with the following naming structure
             "results_{filename}.csv"

    '''
    args = parse_cmdline(argv)

    # Set default numerical arguments
    lin_thresh = args.lin_thresh
    circ_thresh = args.circ_thresh
    frame_rate = args.frame_rate
    σ = args.σ
    ε = args.ε
    video_start_time = args.video_start_time
    fit_type = args.fit_type
    lim = args.lim

    for path in args.path:
      if not os.path.exists(path):
          raise FileNotFoundError(f'Couldn\'t find {path}, '
                                  'make sure you\'ve spelled it right')

    files = []
    for path in args.path:
      if os.path.isfile(path):
          files += [path]
      elif os.path.isdir(args.path):
          exts = '(avi|mp4|jpg|png|gif)'
          if not args.redo:
              files += [args.path+f for f in os.listdir(args.path)
                       if re.match(rf'(?i).*{args.keyword}.*\.{exts}$', f)
                       if not os.path.exists(f'{args.path}results_{f}.csv')]
          else:
              files += [args.path+f for f in os.listdir(args.path)
                       if re.match(rf'(?i).*{args.keyword}.*\.{exts}$', f)]

    plt.ion()
    for file in files:
        # Get the file type for the image file
        print(f'Analyzing {file}')
        ext = os.path.splitext(file)[-1]

        video = False
        if re.match(r'\.(avi|mp4)$', ext):
            video = True
        elif not re.match(r'\.(jpg|png|gif)$', ext):
            raise argparse.ArgumentTypeError(f'Invalid file extension '
                                             f'provided. '
                                             f'I can\'t read {ext} files')

        # At every extracted frame, read it as a grayscale numpy array

        if not video:
            images = [io.imread(file, as_gray=True)]
            time = [0]
        else:
            vid_start = video_start_time
            time, images = extract_grayscale_frames(file,
                                                    start_time=vid_start,
                                                    data_freq=frame_rate)

        # Make sure that the edges are being detected well
        low, high = None, None
        if args.checkFilter:
            σ, low, high = sigma_setter(images[0], σ=σ)

        if not args.auto_crop:
            bounds = get_crop(images[0])
        else:
            bounds = None

        # Define the threshold values in absolute pixel terms
        if circ_thresh <= 0:
          _circ_thresh = -circ_thresh*images[0].shape[0]
        else:
          _circ_thresh = circ_thresh

        if lin_thresh <= 0:
          _lin_thresh = -lin_thresh*images[0].shape[0]
        else:
          _lin_thresh = lin_thresh

        # Run the analysis on each of the frames to be processed
        if args.nproc > 1:
            out = Parallel(n_jobs=args.nproc)(delayed(analyze_frame)(im,
                                                      time[j], bounds,
                                                      _circ_thresh, _lin_thresh,
                                                      σ, low, high,
                                                      ε, lim, fit_type)
                               for j, im in enumerate(images))
        else:
            out = [None]*len(images)
            for j, im in enumerate(images):
                out[j] = analyze_frame(im, time[j], bounds,
                                       _circ_thresh, _lin_thresh, σ,
                                       low, high, ε, lim, fit_type)

        angles, base_width, volumes, fits, baselines = zip(*out)

        if video:
            output_plots(time, angles, base_width, volumes)

        output_fits(images, fits, baselines, bounds,
                    linear=(fit_type=='linear'), savefile=args.savefig)
        output_datafile(file, time, angles, base_width, volumes)

        if args.block_at_end:
            plt.ioff()
            plt.show()
        else:
            sleep(2)
            plt.close('all')
            plt.ioff()
