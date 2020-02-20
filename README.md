# Python contact angle image processing analysis

A python script that follows a simple logical progression to reliably measure contact angles from image or video files. While the traditional methods for contact angle analysis typically rely on the user drawing tangent lines to droplets, which is both time consuming and can lead to bias in the analysis results, we attempt to automate this analysis to make the process both more robust and more amenable to high throughput data generation. The logic we use for this process is highlighted below:

![Logic flow](./images/LogicDiagram.svg)

## Installation

The analysis script can be installed by cloning the repository into your desired working directory or via the following:

```
$ pip install droppy
```

With the `pip` installation, the main script can be run from the command line by calling `droppy`; otherwise it must be run from within a Python instance (see Use section below).

### Dependencies

The following packages must already be installed in your Python environment to contribute to the development of this project:
* numpy
* scipy
* scikit-image
* imageio
* matplotlib
* setuptools
* wheel
* twine
* pytest
* pip:
    * imageio-ffmpeg
    * pytest-subtests
    * pytest-cov

## Use

Depending on the installation choice, the script can either be run from the command line:

```
$ droppy path/to/files/of/interest
```

If you have installed as a developer, you can use the script by calling the `main()` function from the file `analysis.py`

### Parameter Definitions

The relevant threshold parameters that define where the tangent lines, baseline, and circle will be identified are most easily explained through the image below:

![Threshold example image](./images/ThresholdDrawings.svg)

These parameters can be accessed through the flags `--baselineThreshold`, `--circleThreshold`, and `--linThreshold` respectively. Additional flags can be set and can be shown from the help accessed by

```
$ droppy --help
```

### Documentation

A GitHub pages site with the full documentation and API is provided [here](https://michaelorella.github.io/droppy/)


## Credits

Contact angle measurement automation has also been performed by [mvgorcum](https://github.com/mvgorcum/Sessile.drop.analysis), which uses a different approach to fitting the tangents, but inspired our work here.

[<img src="https://avatars0.githubusercontent.com/u/40570716?s=400&u=7bde054e05bbba59c19cefd3aa2f4c84e2a9dfc6&v=4" height="150" width="150">](https://github.com/michaelorella)  [<img src="https://avatars2.githubusercontent.com/u/29216577?s=400&v=4" height="150" width="150">](https://github.com/mcleonard11)

## Contribute

Please don't hesitate to submit any issues that you may identify with the approach or the coding. We will try to respond quickly to any questions that may arise. If you would like to contribute to the project, feel free to make any pull requests that will make the solution more robust/efficient/better for your application, and we will do our best to incorporate it in the next release.
