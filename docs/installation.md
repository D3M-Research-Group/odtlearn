# Installation Guide

This guide explains how to install the ODTlearn Python package.

## Install Python
ODTlearn is a package for [Python](https://www.python.org/). To use ODTlearn first [download and install](https://www.python.org/downloads/) Python. 

```{tip}
We highly recommend using virtual environments with Python. There are many different options for managing Python environments, but we recommend using [Pipenv](https://pipenv.pypa.io/en/latest/).

If you are new to Python, we recommend reading the [Getting started with Python](https://www.python.org/about/gettingstarted/) from the Python Software Foundation.
```

## Install ODTlearn

To install the package, you need to run the following command in your terminal:

```bash
pip install git+https://github.com/D3M-Research-Group/odtlearn.git#egg=odtlearn
```


## Obtain Gurobi License
To use Gurobi with ODTlearn, you must have a valid Gurobi License. [Free licenses are available for academic use](https://www.gurobi.com/academia/academic-program-and-licenses/) and additional methods for obtaining a Gurobi license can be found [here](https://www.gurobi.com/solutions/licensing/).

## CBC Binaries
[Python-MIP](https://github.com/coin-or/python-mip) provides CBC binaries for 64-bit versions of Windows, Linux, and MacOS that run on Intel hardware, however we have observed that these binaries do not seem to work properly with lazy constraint generation, which is used in some of our MIO formulations. Thus, to ensure expected behavior when using ODTlearn, we strongly recommend building CBC from source. Below are the steps needed to compile CBC from source using [coinbrew](https://github.com/coin-or/coinbrew).

```
mkdir CBC
cd CBC
wget -nH https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew 
bash coinbrew fetch Cbc@master --no-prompt
bash coinbrew build Cbc@stable/2.10

export DYLD_LIBRARY_PATH=/PATH/TO/CBC/dist/lib
export PMIP_CBC_LIBRARY=/PATH/TO/CBC/dist/lib/PLATFORM_SPECIFIC_SHARED_LIB
```

The last two steps are critical for ensuring that ODTlearn (through Python-MIP) uses the correct CBC binary. For Windows and MacOS the shared library name is `libCbc.dll` and `libCbc.dylib`, respectively. For Linux, the shared library name is `libCbcSolver.so`. To ensure that the environment variables persist, we suggest adding the last two lines to your `.zshrc` or `.bashrc` file. 

