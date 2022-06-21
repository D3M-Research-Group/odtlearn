#! /usr/bin/env python
"""A package for decision tree methods."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
# ver_file = os.path.join("odtlearn", "_version.py")
# print(ver_file)
# with open(ver_file) as f:
#     exec(f.read())

DISTNAME = "odtlearn"
DESCRIPTION = "A package for tree-based statistical estimation and inference using optimal decision trees."
with codecs.open("README.md", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = "Sina Aghaei, Nathanael Jo, Nathan Justin, Patrick Vossler"
MAINTAINER = "Patrick Vossler"
MAINTAINER_EMAIL = "patrick.vossler18@gmail.com"
URL = "https://github.com/D3M-Research-Group/odtlearn"
LICENSE = "GPL-3"
DOWNLOAD_URL = "https://github.com/D3M-Research-Group/odtlearn"
VERSION = "0.0.1"
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn", "pandas", "gurobipy"]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.9",
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
    packages=find_packages(include=["odtlearn", "odtlearn.*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    package_data={"": ['data/*.csv', 'data/*.npz']}
)
