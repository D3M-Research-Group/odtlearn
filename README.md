# ODTlearn

A package for tree-based statistical estimation and inference using optimal decision trees. ODTlearn provides implementations of StrongTrees ([1]), FairTrees ([2]), RobustTrees ([3]) for classification, and Prescriptive Trees ([4]) for prescription.

## Documentation

The [package documentation](https://d3m-research-group.github.io/odtlearn/index.html) contains usage examples and method reference.

## Installation

The current development version can be installed from source with the following command:

``` bash
pip install git+https://github.com/D3M-Research-Group/odtlearn.git#egg=odtlearn
```

A release version of the package will be available on PyPI shortly.

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