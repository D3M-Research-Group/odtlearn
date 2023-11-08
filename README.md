<p align="center">
<img src="./img/ODTlearn-color.png" alt="ODTlearn Logo" width="500"/>
</p>

A package for tree-based statistical estimation and inference using optimal decision trees. ODTlearn provides implementations of StrongTrees [1], FairTrees [2], RobustTrees [3] for classification, and Prescriptive Trees [4] for prescription.

![Test badge](https://github.com/D3M-Research-Group/odtlearn/actions/workflows/ci.yml/badge.svg)
![Documentation badge](https://github.com/D3M-Research-Group/odtlearn/actions/workflows/sphinx.yml/badge.svg)
![License badge](https://img.shields.io/github/license/D3M-Research-Group/odtlearn)


## Documentation

The [package documentation](https://d3m-research-group.github.io/odtlearn/index.html) contains usage examples and method reference.

## Installation

The current development version can be installed from source with the following command:

``` bash
pip install git+https://github.com/D3M-Research-Group/odtlearn.git#egg=odtlearn
```

A release version of the package will be available on PyPI shortly.

### Obtain Gurobi License
To use Gurobi with ODTlearn, you must have a valid Gurobi License. [Free licenses are available for academic use](https://www.gurobi.com/academia/academic-program-and-licenses/) and additional methods for obtaining a Gurobi license can be found [here](https://www.gurobi.com/solutions/licensing/).

### CBC Binaries
[Python-MIP](https://github.com/coin-or/python-mip) provides CBC binaries for 64-bit versions of Windows, Linux, and MacOS that run on Intel hardware, however we have observed that these binaries do not seem to work properly with lazy constraint generation, which is used in some of our MIO formulations. Thus, to ensure expected behavior when using ODTlearn, we strongly recommend building CBC from source. Below are the steps needed to compile CBC from source using [coinbrew](https://github.com/coin-or/coinbrew).

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
* [1] Aghaei, S., Gómez, A., & Vayanos, P. (2021). Strong optimal classification trees. arXiv preprint arXiv:2103.15965. https://arxiv.org/abs/2103.15965.
* [2] Jo, N., Aghaei, S., Benson, J., Gómez, A., & Vayanos, P. (2022). Learning optimal fair classification trees. arXiv preprint arXiv:2201.09932. https://arxiv.org/pdf/2201.09932.pdf
* [3] Justin, N., Aghaei, S., Gomez, A., & Vayanos, P. (2021). Optimal Robust Classification Trees. In The AAAI-22 Workshop on Adversarial Machine Learning and Beyond. https://openreview.net/pdf?id=HbasA9ysA3
* [4] Jo, N., Aghaei, S., Gómez, A., & Vayanos, P. (2021). Learning optimal prescriptive trees from observational data. arXiv preprint arXiv:2108.13628. https://arxiv.org/pdf/2108.13628.pdf

