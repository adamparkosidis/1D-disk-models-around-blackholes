"""
Setup file for package `disklab`.
"""
from numpy.distutils.core import Extension, setup
from setuptools import find_packages
import os
import warnings

diffusion = Extension(
    name='disklab.diffusion',
    sources=['disklab/diffusion.f90'])

extensions = [diffusion]


def setup_function(extensions):
    setup(name='disklab',
          description='circumstellar disk model and helper routines',
          long_description=open(os.path.join(
              os.path.dirname(__file__), 'README.md')).read(),
          url='https://github.com/dullemond/DISKLAB',
          author='Kees Dullemond & Til Birnstiel',
          author_email='dullemond_AT_uni-heidelberg.de & til.birnstiel_AT_lmu.de',
          license='GPLv3',
          packages=find_packages(),
          include_package_data=True,
          package_data={'disklab': [
                        'disklab/opacity/precalculated/*',
                        'disklab/opacity/optical_constants/*/*',
                        'disklab/opacity/optical_constants/*/*/*']},
          install_requires=['scipy', 'numpy', 'matplotlib'],
          zip_safe=False,
          ext_modules=extensions,
          )


try:
    setup_function(extensions)
except BaseException:
    warnings.warn('Failed building the 2D extensions - will install DISKLAB without them. See users guide for installing them manually.')
    warnings.warn('Mie calculations will use python routines -- this will be much slower than compiled code.')
    setup_function([])
