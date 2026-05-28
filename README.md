<p align="center">
<img src="./img/ODTlearn-color.png" alt="ODTlearn Logo" width="500"/>
</p>

A package for tree-based statistical estimation and inference using optimal decision trees. ODTlearn provides implementations of StrongTrees [1], FairTrees [2], RobustTrees [3,4] for classification, and Prescriptive Trees [5] for prescription.

![Test badge](https://github.com/D3M-Research-Group/odtlearn/actions/workflows/ci.yml/badge.svg)
![Documentation badge](https://github.com/D3M-Research-Group/odtlearn/actions/workflows/sphinx.yml/badge.svg)
![License badge](https://img.shields.io/github/license/D3M-Research-Group/odtlearn)


## Documentation

The [package documentation](https://d3m-research-group.github.io/odtlearn/index.html) contains usage examples and method reference.

## Installation
ODTlearn is a package for [Python](https://www.python.org/). To use ODTlearn first [download and install](https://www.python.org/downloads/) Python. ODTLearn supports Python 3.9 and later. CBC solver support is only available for Python 3.9-3.11.

The latest stable version can be installed from PyPI with the command:

``` bash
pip install odtlearn
```

The current development version can be installed from source with the following command:

``` bash
pip install git+https://github.com/D3M-Research-Group/odtlearn.git
```

This package gives you the option to either use Gurobi or Coin-OR Branch & Cut (CBC) as your base solver.

### Using Gurobi
To use Gurobi with ODTlearn, you must have a valid Gurobi License. [Free licenses are available for academic use](https://www.gurobi.com/academia/academic-program-and-licenses/) and additional methods for obtaining a Gurobi license can be found [here](https://www.gurobi.com/solutions/licensing/).

Once a license is obtained, you must install the `gurobipy` package:
```bash
pip install gurobipy
```

### Using CBC
Note that currently, CBC is only supported in ODTLearn for Python 3.9-3.11.

First, install [Python-MIP](https://github.com/coin-or/python-mip) and [CFFI](https://cffi.readthedocs.io/en/stable/index.html):
```bash
pip install mip cffi
```

Python-MIP provides CBC binaries for 64-bit versions of Windows, Linux, and MacOS that run on Intel hardware, however we have observed that these binaries do not seem to work properly with lazy constraint generation, which is used in some of our MIO formulations. Thus, to ensure expected behavior when using ODTlearn, we strongly recommend building CBC from source. Below are the steps needed to compile CBC from source using [coinbrew](https://github.com/coin-or/coinbrew).


``` bash
mkdir CBC
cd CBC
wget -nH https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew 
bash coinbrew fetch Cbc@master --no-prompt
bash coinbrew build Cbc@stable/2.10

export DYLD_LIBRARY_PATH=/PATH/TO/CBC/dist/lib
export PMIP_CBC_LIBRARY=/PATH/TO/CBC/dist/lib/PLATFORM_SPECIFIC_SHARED_LIB
```

The last two steps are critical for ensuring that ODTlearn (through Python-MIP) uses the correct CBC binary. For Windows and MacOS the shared library name is `libCbc.dll` and `libCbc.dylib`, respectively. For Linux, the shared library name is `libCbc.so`. To ensure that the environment variables persist, we suggest adding the last two lines to your `.zshrc` or `.bashrc` file. 



## Developing
This project uses ``black`` to format code and ``flake8`` for linting. We also support ``pre-commit`` to ensure
these have been run. To configure your local environment please install these development dependencies and set up
the commit hooks.

``` bash
pip install black flake8 pre-commit
pre-commit install
```


## References
* [1] Aghaei, S., Gómez, A., & Vayanos, P. (2025). Strong optimal classification trees. *Operations Research*, 73(4), 2223-2241.
* [2] Jo, N., Aghaei, S., Benson, J., Gómez, A., & Vayanos, P. (2022). Learning optimal fair classification trees. *arXiv preprint* arXiv:2201.09932. https://arxiv.org/pdf/2201.09932.pdf
* [3] Justin, N., Aghaei, S., Gomez, A., & Vayanos, P. (2021). Optimal Robust Classification Trees. In *The AAAI-22 Workshop on Adversarial Machine Learning and Beyond*. https://openreview.net/pdf?id=HbasA9ysA3
* [4] Justin, N., Aghaei, S., Gómez, A., & Vayanos, P. (2023). Learning optimal classification trees robust to distribution shifts. *arXiv preprint* arXiv:2310.17772.
* [5] Jo, N., Aghaei, S., Gómez, A., & Vayanos, P. (2021). Learning optimal prescriptive trees from observational data. *arXiv preprint* arXiv:2108.13628. https://arxiv.org/pdf/2108.13628.pdf

