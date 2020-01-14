import sys
import os

import matplotlib.pyplot as plt

import numpy as np

import re

from skimage import io

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
                                         find_intersection)

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

    if os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        files = [args.path+f for f in os.listdir(args.path)
                 if re.match(r'.*\.(avi|mp4|jpg|png|gif)$', f)
                 if not os.path.exists(f'results_{f}.csv')]

    plt.ion()
    for file in files:
        # Get the file type for the image file
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

        bounds = get_crop(images[0])

        angles = []
        volumes = []
        base_width = []

        # Make sure that the edges are being detected well
        low, high = None, None
        if args.checkFilter:
            σ, low, high = sigma_setter(images[0], σ=σ, bounds=bounds)

        # Create a set of axes to hold the scatter points for all frames in
        # the videos
        plt.figure()
        scatter_axis = plt.axes()

        plt.figure(figsize=(5, 5))
        image_axes = plt.axes()
        plt.show()

        for j, im in enumerate(images):
            coords = extract_edges(im, σ=σ, low=low, high=high)
            crop = crop_points(coords, bounds)

            # Get the baseline from the left and right threshold pixels of the
            # image (this is important not to crop too far)
            baseline = {L: crop_points(coords, [bounds[0],
                                                bounds[0]+baseline_threshold,
                                                *bounds[2:]]),
                        R: crop_points(coords, [bounds[1]-baseline_threshold,
                                                bounds[1],
                                                *bounds[2:]])}

            a = fit_line(np.concatenate((baseline[L],
                                         baseline[R])), base_ord)[0]

            f = {i: lambda x, y: x for i in [L, R]}
            f[B] = lambda x, y: y - (np.dot(a, np.power(x, range(len(a)))))
            f[T] = lambda x, y: y

            b = np.copy(bounds)
            b[3] = - circ_thresh
            circle = crop_points(crop, b, f=f)

            # Make sure that flat droplets (wetted) are ignored
            # (i.e. assign angle to NaN and continue)
            if circle.shape[0] < 5:
                angles += [(np.NaN, np.NaN)]
                base_width += [np.NaN]
                continue

            scatter_axis.scatter(circle[:, 0], circle[:, 1])
            scatter_axis.invert_yaxis()

            # Plot the current image
            image_axes.clear()
            image_axes.imshow(im, cmap='gray', vmin=0, vmax=1)
            image_axes.axis('off')

            # Baseline
            x = np.linspace(0, im.shape[1])
            y = np.dot(a, np.power(x, [[po]*len(x)
                                       for po in range(base_ord + 1)]))
            image_axes.plot(x, y, 'r-')
            plt.show()

            if fit_type == 'linear':
                b = np.copy(bounds)
                b[3] = -(circ_thresh + lin_thresh)
                limits = generate_droplet_width(crop, b, f)

                # Get linear points
                f[T] = f[B]
                linear_points = {L: crop_points(crop,
                                                [int(limits[L]-lin_thresh/2),
                                                 int(limits[L]+lin_thresh/2),
                                                 -(circ_thresh+lin_thresh),
                                                 -circ_thresh], f=f),
                                 R: crop_points(crop,
                                                [int(limits[R]-lin_thresh/2),
                                                 int(limits[R]+lin_thresh/2),
                                                 -(circ_thresh+lin_thresh),
                                                 -circ_thresh], f=f)}

                v, b, m, bv, vertical = generate_vectors(linear_points,
                                                         limits, ε, a)

                # Calculate the angle between these two vectors defining the
                # base-line and tangent-line
                ϕ = {i: calculate_angle(bv[i], v[i]) for i in [L, R]}

                # Plot lines
                for side in [L, R]:
                    x = np.linspace(0, im.shape[1])
                    if not vertical[side]:
                        y = m[side] * x + b[side]
                    else:
                        y = np.linspace(0, im.shape[0])
                        x = m[side] * y + b[side]
                    image_axes.plot(x, y, 'r-')

                baseline_width = limits[R] - limits[L]

                volume = np.NaN
                # TODO:// Add the actual volume calculation here!

            elif fit_type == 'circular':
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
                baseline_width = 2 * x_t

                volume = (2/3 * np.pi * r ** 3
                          + np.pi * r ** 2 * y_t
                          - np.pi * y_t ** 3 / 3)

                # Fitted circle
                theta = np.linspace(0, 2 * np.pi, num=100)
                x = z[0] + r * np.cos(theta)
                y = z[1] + r * np.sin(theta)
                image_axes.plot(x, y, 'r-')

            else:
                raise Exception('Unknown fit type! Try another.')

            # FI FITTYPE
            output_text(time[j], ϕ, baseline_width, volume)
            angles += [(ϕ[L], ϕ[R])]
            base_width += [baseline_width]
            volumes += [volume]

            # Format the plot nicely
            image_axes.set_xlim(bounds[0:2])
            image_axes.set_ylim(bounds[-1:-3:-1])
            plt.draw()
            plt.pause(0.1)

        # END LOOP THROUGH IMAGES

        if video:
            output_plots(time, angles, base_width, volumes)

        output_datafile(file, time, angles, base_width, volumes)
    if args.block_at_end:
        plt.ioff()
        plt.show()
    else:
        plt.close('all')
        plt.ioff()

if __name__ == '__main__':
    main()
