# ODTlearn

A package for tree-based statistical estimation and inference using optimal decision trees. ODTlearn provides implementations of StrongTrees [1], FairTrees [2], RobustTrees [3] for classification, and Prescriptive Trees [4] for prescription.

![License](https://img.shields.io/github/license/D3M-Research-Group/odtlearn)


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
[Python-MIP](https://github.com/coin-or/python-mip) provides CBC binaries for 64-bit versions of Windows, Linux, and MacOS that run on Intel hardware. 

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