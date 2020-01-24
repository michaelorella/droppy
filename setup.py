from setuptools import setup
import versioneer

with open('README.md', 'r') as f:
    desc = f.read()

setup(name="contactangles",
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author="Michael J. Orella",
      author_email="morella@mit.edu",
      license='MIT',
      packages=['contactangles'],
      python_requires='>3.6',
      test_suite='tests',
      install_requires=['numpy',
                        'scipy',
                        'scikit-image',
                        'imageio',
                        'matplotlib',
                        'imageio-ffmpeg',
                        'pyqt5',
                        'numba'],
      description=desc.splitlines()[0],
      long_description=desc,
      long_description_content_type='text/markdown',
      url='https://github.com/michaelorella/contactangles',
      entry_points={'console_scripts':
                    ['analysis=contactangles.analysis:main']
                    }
      )
