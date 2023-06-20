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
Python-MIP provides CBC binaries for 64-bit versions of Windows, Linux, and MacOS that run on Intel hardware. 

If you are using an Apple computer with an M1 processor, you will need to compile CBC from source. Below are the steps needed to compile CBC from source using [coinbrew](https://github.com/coin-or/coinbrew) and then copy the resulting dynamic library to the folder containing your python-mip libraries.

```
mkdir CBC
cd CBC
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew 
./coinbrew fetch Cbc@master
./coinbrew build Cbc 
cp dist/lib/libCbc.0.dylib /PATH/TO/YOUR/PYTHON/INSTALLATION/lib/python3.9/site-packages/mip/libraries/cbc-c-darwin-x86-64.dylib
```

Tips for determining the path to your Python installation's site-packages folder can be found in [this stackoverflow question](https://stackoverflow.com/questions/122327/how-do-i-find-the-location-of-my-python-site-packages-directory).
