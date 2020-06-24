__author__ = "Michael Orella"
__email__ = "morella@mit.edu"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import sys

if 'main' not in sys.modules:
	from .droppy import main

def run(*args, **kwargs):
	_params = []
	for param in args:
		_params += param

	for key in kwargs:
		_params += [key]
		_params += [kwargs[key]]
	
	main(_params)


