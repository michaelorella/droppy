import skimage
from skimage import feature
from skimage.viewer import plugins
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider, ComboBox

import matplotlib.pyplot as plt

import numpy as np

def sigma_setter(image, σ=1, bounds=None):
    '''
    Show the user the image with the detected edges overlain in a way that
    they can update the edge detection parameters and see the impact on the
    edges.

    :param image: 2D numpy grayscale image
    :param σ: Standard deviation value for the Gaussian blur applied before
              edge detection
    :param bounds: List of [left, right, top, bottom] crop for the image area
    '''
    plt.ioff()
    # edges = feature.canny(image, sigma=σ)
    viewer = ImageViewer(image)
    if bounds is not None:
        viewer.ax.set_xlim(bounds[0:2])
        viewer.ax.set_ylim(bounds[-1:-3:-1])

    canny = CannyPlugin()
    viewer += canny
    out = viewer.show()[0][1]

    σ = out['sigma']
    low = out['low_threshold']
    high = out['high_threshold']

    print(f'Proceeding with sigma = {σ : 6.2f}')
    plt.ion()
    return σ, low, high


def extract_edges(image, σ=1, low=None, high=None, indices=True):
    '''
    Compute the detected edges using the canny algorithm

    :param image: numpy grayscale image
    :param σ: canny filter value to use
    :param check_σ: flag whether to visually check the edges that are detected
    :return: list of [x,y] coordinates for the detected edges
    '''
    edges = feature.canny(image, sigma=σ, low_threshold=low,
                          high_threshold=high)
    if not indices:
        return edges
    return np.argwhere(edges.T)


class CannyPlugin(plugins.OverlayPlugin):
    '''
    Modification of Canny Plugin provided by Scikit-image to return parameter
    values

    Uses the default values of:

    - sigma == 0 (No Gaussian blurring)
    - high == 1 (strong pixels are all the way on)
    - low == 1 (weak pixels are all the way on)

    '''

    name = 'Canny Filter'

    def __init__(self, *args, **kwargs):
        super().__init__(image_filter=feature.canny, **kwargs)
        self.sigma = 0
        self.low_threshold = 1
        self.high_threshold = 1

    def add_widget(self, widget):
        '''
        Add the new Canny filter widget ot the plugin while registering a new
        callback method to set the widgets attributes so they can be accessed
        later.
        '''
        super().add_widget(widget)

        if widget.ptype == 'kwarg':
            def update(*widget_args):
                setattr(self, widget.name, widget.val)
                self.filter_image(*widget_args)

            widget.callback = update

    def attach(self, image_viewer):
        '''
        Override the attaching of the plugin to the ImageViewer. This utilizes
        nearly identical implementation to https://github.com/scikit-image/
        scikit-image/blob/master/skimage/viewer/plugins/canny.py, but changes
        the limits for parameter selection.
        '''
        image = image_viewer.image
        imin, imax = skimage.dtype_limits(image, clip_negative=False)
        itype = 'float' if np.issubdtype(image.dtype, np.floating) else 'int'
        self.add_widget(Slider('sigma', 0, 2, update_on='release'))
        self.add_widget(Slider('low_threshold', imin, imax, value_type=itype,
                        update_on='release'))
        self.add_widget(Slider('high_threshold', imin, imax, value_type=itype,
                        update_on='release'))
        self.add_widget(ComboBox('color', self.color_names, ptype='plugin'))
        # Call parent method at end b/c it calls `filter_image`, which needs
        # the values specified by the widgets. Alternatively, move call to
        # parent method to beginning and add a call to `self.filter_image()`
        super(CannyPlugin,self).attach(image_viewer)

    def output(self):
        '''
        Override the default output behavior so that when the ImageViewer is
        closed, the result contains parameter values that we need to pass on
        to all future edge detection calls.
        '''
        new = (super().output()[0], {'sigma': self.sigma,
                                     'low_threshold': self.low_threshold,
                                     'high_threshold': self.high_threshold})
        return new
