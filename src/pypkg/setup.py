#!/usr/bin/env python

"""
setup.py file for SWIG interface
"""

from setuptools import setup


setup (name = 'pymgpg',
       version = '0.2',
       author      = "Marco Virgolin",
       packages=['pymgpg'],
	package_data={'pymgpg': ['_pb_mgpg.so']},
       include_package_data=True,
       install_requires=['scikit-learn', 'sympy'],
       )