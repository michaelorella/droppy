import sys

# if __name__ == '__main__':
#     sys.path = [''] + sys.path

import os

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

import numpy as np

import re

from skimage import io

import time as t

from contactangles.imageanalysis import (L, R, T, B)
from contactangles.imageanalysis import (parse_cmdline,
                                         calculate_angle,
                                         baseF,
                                         crop_points,
                                         get_crop,
                                         fit_line,
                                         extract_edges,
                                         generate_vectors,
                                         generate_droplet_width,
                                         output_text,
                                         dist,
                                         sigma_setter,
                                         fit_circle,
                                         generate_circle_vectors,
                                         find_intersection,
                                         fit_bashforth_adams,
                                         sim_bashforth_adams,
                                         analyze_frame,
                                         output_fits,
                                         auto_crop)

from contactangles.movie_handling import (extract_grayscale_frames,
                                          output_plots,
                                          output_datafile)


def main(argv=None):
    args = parse_cmdline(argv)

    # Set default numerical arguments
    baseline_threshold = args.baseline_threshold
    lin_thresh = args.lin_thresh
    circ_thresh = args.circ_thresh
    frame_rate = args.frame_rate
    base_ord = args.base_ord
    σ = args.σ
    ε = args.ε
    video_start_time = args.video_start_time
    tolerance = args.tolerance
    fit_type = args.fit_type
    lim = args.lim

    if not os.path.exists(args.path):
        raise FileNotFoundError(f'Couldn\'t find {args.path}, '
                                'make sure you\'ve spelled it right')

    if os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        exts = '(avi|mp4|jpg|png|gif)'
        if not args.redo:
            files = [args.path+f for f in os.listdir(args.path)
                     if re.match(rf'(?i).*{args.keyword}.*\.{exts}$', f)
                     if not os.path.exists(f'{args.path}results_{f}.csv')]
        else:
            files = [args.path+f for f in os.listdir(args.path)
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
            bounds = auto_crop(images[0])

        # # Create a set of axes to hold the scatter points for all frames in
        # # the videos
        # plt.figure()
        # scatter_axis = plt.axes()
        # scatter_axis.invert_yaxis()

        # plt.figure(figsize=(5, 5))
        # image_axes = plt.axes()
        # plt.show()

        out = Parallel(n_jobs=8)(delayed(analyze_frame)(im, time[j], bounds,
                                                  baseline_threshold,
                                                  circ_thresh, lin_thresh,
                                                  base_ord, σ, low, high,
                                                  ε, lim, fit_type)
                           for j, im in enumerate(images))

        angles, base_width, volumes, fits, baselines = zip(*out)

        if video:
            output_plots(time, angles, base_width, volumes)

        output_fits(images, fits, baselines, bounds)
        output_datafile(file, time, angles, base_width, volumes)

        if args.block_at_end:
            plt.ioff()
            plt.show()
        else:
            t.sleep(2)
            plt.close('all')
            plt.ioff()

if __name__ == '__main__':
    main()
