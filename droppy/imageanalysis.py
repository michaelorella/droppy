from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool

import matplotlib.pyplot as plt

import numpy as np

import warnings

from droppy.common import L, R, T, B

def get_crop(image):
    '''
    Show the original image to allow the user to crop any extraneous
    information out of the frame.

    :param image: 2D numpy array grayscale image
    :return: list of [left, right, top, bottom] values for the edges of the
             bounding box
    '''
    plt.ioff()
    print('Waiting for input, please crop the image as desired and hit enter')
    viewer = ImageViewer(image)
    rect_tool = RectangleTool(viewer, on_enter=viewer.closeEvent)
    viewer.show()

    bounds = np.array(rect_tool.extents)
    if (bounds[0] < 0 or bounds[1] > image.shape[1] or bounds[2] < 0
            or bounds[3] > image.shape[0]):
        print(f'Bounds = {bounds} and shape = {image.shape}')
        raise ValueError('Bounds outside image, make sure to select within')

    plt.ion()
    return np.array(np.round(bounds), dtype=int)

def auto_crop(image, pad=50, σ=1, low=None, high=None):
    '''
    Automatically identify where the crop should be placed within the original
    image

    This function makes several assumptions about the image layout to look
    at where the edge density is the highest and center the crop at these
    locations. The bounds of the image are typically identified by looking
    for sharp changes in the edge density as a function of x or y position.

    :param image: 2D numpy array of [x,y] coordinates of the edges of the
                  image
    :param pad: width in pixels of space around the automatically identified
                cropping box
    :param σ: Gaussian filter used for the Canny detection algorithm
    :param low: Value of the weak pixels in the dual thresholding
    :param high: value of the strong pixels in the dual thresholding
    :return: list of [left, right, top, bottom] values for the edges of the
             bounding box
    '''
    print('Performing auto-cropping, please wait...')
    edges = extract_edges(image, σ=σ, low=low, high=high)
    x = edges[:, 0]
    y = edges[:, 1]

    binwidth = 1
    lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    bins = np.arange(0, lim + 2*binwidth, binwidth)

    hist, bin_edges = np.histogram(y, bins=bins)
    zero_locations = [b for b, c in zip(bin_edges, hist) if c == 0]
    tops = [b for b, δ in zip(zero_locations, np.diff(zero_locations))
            if δ > 1]
    zero_loc, top, *_ = tops
    bottom = zero_locations[zero_locations.index(top) + 1] + pad
    top -= pad

    temp_x = np.array([x_pt for x_pt, y_pt in zip(x, y)
                       if top <= y_pt <= bottom])
    hist, bin_edges = np.histogram(temp_x, bins=bins)
    temp_x = np.array([x_pt for x_pt in temp_x
                       if hist[np.where(bin_edges==x_pt)] > 1])
    nonzeros = np.array([b for c, b in zip(*np.histogram(temp_x, bins=bins))
                         if c != 0])
    gaps = np.where(np.diff(nonzeros) > pad)[0]

    if gaps.size != 0:
        #Have some large gap
        x_bar = nonzeros.mean()
        left_sides = nonzeros[gaps]
        if (left_sides < x_bar).any():
            left = nonzeros[np.where(nonzeros == left_sides
                                     [np.where(left_sides
                                               < x_bar)][-1])[0][0] + 1]
        else:
            left = nonzeros[0]

        left -= pad

        if (left_sides > x_bar).any():
            right = nonzeros[np.where(nonzeros == left_sides
                                     [np.where(left_sides
                                               > x_bar)][0])[0][0]]
        else:
            right = nonzeros[-1]

        right += pad
    else:
        left, right = nonzeros[0] - pad, nonzeros[-1] + pad

    bounds = [left, right, top, bottom]
    return np.array(np.round(bounds), dtype=int)

def crop_points(image, bounds, f=None):
    '''
    Return a cropped image with points only lying inside the bounds

    :param image: numpy array of [x,y] coordinates of detected edges
    :param bounds: list of [left, right, top, bottom] image bounds - because
                   of the way that images are read, bottom > top numerically
    :return: abbreviated numpy array of [x,y] coordinates that are within
             bounds
    '''

    if f is None:
        f = {i: lambda x, y: x for i in [L, R]}
        f[T] = lambda x, y: y
        f[B] = lambda x, y: y

        if bounds[2] > bounds[3]:
            warnings.warn('Check the order of the bounds, as bottom > top')

        if any([x < 0 for x in bounds]):
            warnings.warn('All bounds must be positive')

    new_im = np.array([[x,y] for x, y in image if (f[L](x, y) >= bounds[0] and
                                                   f[R](x, y) <= bounds[1] and
                                                   f[T](x, y) >= bounds[2] and
                                                   f[B](x, y) <= bounds[3])])

    return new_im

def output_text(time, φ, baseline_width, volume):
    '''
    Show the results of the frame analyzed on the command line

    >>> output_text(1.0, (90, 90), 100, 100000)
    At time 1.0 Contact angle left (deg): 90 Contact angle right (deg): 90
    contact angle average (deg): 90 Baseline width (px): 100

    :param time: float time in seconds to be reported
    :param ϕ: tuple of contact angles at the left and right droplet edges
    :param baseline_width: float distance in pixels between the contact points
                           of the droplet
    :param volume: volume of the droplet in pixels**3
    :return: None, outputs a line of text to the terminal
    '''
    print(f'At time {time : 6.3f}: \t'
          f' Contact angle left (deg): {ϕ[L] : 6.3f} \t'
          f' Contact angle right (deg): {ϕ[R] : 6.3f} \t'
          f' Contact angle average (deg): {(ϕ[L]+ϕ[R])/2 : 6.3f} \t'
          f' Baseline width (px): {baseline_width : 4.1f}')

def output_fits(images, fits, baselines, bounds, linear=False):
    '''
    Plot the original images with the overlaid fits

    :param images: List of 2D numpy grayscale images
    :param fits: List of 2D numpy [x,y] coordinates that specify the fit
                 locations
    :param baselines: List of 2D numpy [x,y] coordinates that specify baseline
    :param bounds: list of [left, right, top, bottom]
    :param linear: boolean to tell whether the fittype was linear or not
    :return: None outputs plots comparing the original images to the best fits
    '''
    num = int(np.ceil(np.sqrt(len(images))))
    fig, axes = plt.subplots(nrows=num, ncols=num)
    axes = axes.flatten()
    for j, im in enumerate(images):
        axes[j].imshow(im, cmap='gray', vmin=0, vmax=1)
        axes[j].axis('off')
        axes[j].plot(baselines[j][:, 0], baselines[j][:, 1], 'r-')
        if linear:
            axes[j].plot(fits[j][L][:, 0], fits[j][L][:, 1], 'r-')
            axes[j].plot(fits[j][R][:, 0], fits[j][R][:, 1], 'r-')
        else:
            axes[j].plot(fits[j][:, 0], fits[j][:, 1], 'r-')
        axes[j].set_xlim(bounds[0:2])
        axes[j].set_ylim(bounds[-1:-3:-1])

    for k in range(j, num**2):
        axes[k].axis('off')

    for k in range(j, num**2):
        axes[k].axis('off')

    plt.figure(figsize=(1.5,1.5))
    plt.imshow(images[-1], cmap='gray', vmin=0, vmax=1)
    plt.plot(baselines[-1][:, 0], baselines[-1][:, 1], 'r-')
    plt.axis('off')
    if linear:
        plt.plot(fits[-1][L][:, 0], fits[j][L][:, 1], 'r-')
        plt.plot(fits[-1][R][:, 0], fits[j][R][:, 1], 'r-')
    else:
        plt.plot(fits[-1][:, 0], fits[j][:, 1], 'r-')
    plt.ylim(bounds[-1:-3:-1])
    plt.xlim(bounds[0:2])

    plt.show()
