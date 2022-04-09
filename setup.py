#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = 'PANet'
DESCRIPTION = 'Dictionary class with advanced functionality'
EMAIL = 'b123767195@gmail.com'
AUTHOR = 'ShuYuHuang'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'
REQUIRED = [
    # 'requests', 'maya', 'records',
]

EXTRAS = {
    # 'fancy feature': ['django'],
}

here = os.path.abspath(os.path.dirname(__file__))
    
if __name__ == '__main__':
    setup(
        name=NAME,
        python_requires=REQUIRES_PYTHON,
        packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"])
    )