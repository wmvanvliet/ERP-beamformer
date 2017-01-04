#! /usr/bin/env python
from setuptools import setup
import os

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name='erp_beamformer',
          maintainer='Marijn van Vliet',
          maintainer_email='w.m.vanvliet@gmail.com',
          description='Construct and apply ERP beamformer filters',
          license='BSD-2',
          url='https://github.com/wmvanvliet/ERP-beamformer',
          version='0.1',
          download_url='https://github.com/wmvanvliet/ERP-beamformer',
          long_description=open('README.md').read(),
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=['erp_beamformer'],
      )
