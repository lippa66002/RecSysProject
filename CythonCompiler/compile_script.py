#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/07/2017

@author: Maurizio Ferrari Dacrema
"""

"""
This script is called in a subprocess and compiles the cython source file provided

python compile_script.py filename.pyx build_ext --inplace
"""

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy, sys, re, os

if len(sys.argv) != 4:
    raise ValueError("Wrong number of parameters received. Expected 4, got {}".format(sys.argv))

# Get the name of the file to compile
fileToCompile = sys.argv[1]

# Remove the argument from sys argv in order for it to contain only what setup needs
del sys.argv[1]

# Convert file path to module name by replacing slashes with dots and removing the .pyx extension
extensionName = re.sub(r'\.pyx$', '', fileToCompile.replace(os.path.sep, '.').replace('/', '.').replace('\\', '.'))

ext_modules = Extension(extensionName,
                [fileToCompile],
                extra_compile_args=['-O2'],
                include_dirs=[numpy.get_include(),],
                )

ext_modules.cython_directives = {'language_level': '3'}

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_modules]
)