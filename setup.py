#!/usr/bin/env python

from distutils.core import setup

setup(name='KeywordSpotter',
      version='1.0',
      description='Keyword recognition',
      author='Luciano Lorenti',
      author_email='lucianolorenti@gmail.com',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['keyword_spotting'],
      install_requires=[
          'opencv-python',
          'pandas',
          'tqdm',
          'scipy',
          'scikit-learn'
      ],
      )
