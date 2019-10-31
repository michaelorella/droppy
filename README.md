# Python contact angle image processing analysis

A python script that follows a simple logical progression to reliably measure contact angles from image or video files. While the traditional methods for contact angle analysis typically rely on the user drawing tangent lines to droplets, which is both time consuming and can lead to bias in the analysis results, we attempt to automate this analysis to make the process both more robust and more amenable to high throughput data generation. The logic we use for this process is highlighted below:

![Logic flow](./LogicDiagram.svg)

## Installation

The analysis script can be installed by cloning the repository into your desired working directory. 

### Dependencies

The following packages must already be installed for your Python installation:
* numpy
* scipy
* scikit-image
* imageio
* matplotlib
* imageio-ffmpeg (must be pip-installed)

## Use

The script can be run from the command line with several arguments that modify the behavior of the tangent fitting. 

```
$ python analysis.py 'your_filename_here.(avi|png|jpg)' -ss 10
```

### Parameter Definitions

The relevant threshold parameters that define where the tangent lines, baseline, and circle will be identified are most easily explained through the image below:

![Threshold example image](./ThresholdDrawings.svg)

These parameters can be accessed through the flags `--baselineThreshold`, `--circleThreshold`, and `--linThreshold` respectively. Additional flags can be set including:

`--baselineOrder` : The order of the polynomial that is used to fit the baseline \
`--startSeconds` : The amount of time (in seconds) that is used to "burn in" the video file before consistent drops are reached \
`--times` : How frequently (in seconds) the frames should be analyzed \
`-s` : The initial filter value that should be used for the edge-detection algorithm

## Credits

Contact angle measurement automation has also been performed by [mvgorcum](https://github.com/mvgorcum/Sessile.drop.analysis), which uses a different approach to fitting the tangents, but inspired our work here.

[<img src="https://avatars0.githubusercontent.com/u/40570716?s=400&u=7bde054e05bbba59c19cefd3aa2f4c84e2a9dfc6&v=4" height="150" width="150">](https://github.com/michaelorella)  [<img src="https://avatars2.githubusercontent.com/u/29216577?s=400&v=4" height="150" width="150">](https://github.com/mcleonard11)

## Contribute

Please don't hesitate to submit any issues that you may identify with the approach or the coding. We will try to respond quickly to any questions that may arise. If you would like to contribute to the project, feel free to make any pull requests that will make the solution more robust/efficient/better for your application, and we will do our best to incorporate it in the next release.