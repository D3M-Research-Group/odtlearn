#! /usr/bin/env python
"""A package for decision tree methods."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
# ver_file = os.path.join("trees", "_version.py")
# print(ver_file)
# with open(ver_file) as f:
#     exec(f.read())

DISTNAME = "trees"
DESCRIPTION = "A package for decision tree methods."
with codecs.open("README.md", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "P. Vossler"
MAINTAINER_EMAIL = "patrick.vossler18@gmail.com"
URL = "https://github.com/scikit-learn-contrib/project-template"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/scikit-learn-contrib/project-template"
VERSION = "0.0.1"
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn"]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(include=["trees", "trees.*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
