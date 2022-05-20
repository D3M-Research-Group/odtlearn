% odtlearn documentation master file

# Introduction

Welcome to odtlearn's documentation!

## What is ODTlearn?
Decision trees are among the most popular and inherently interpretable machine learning models and are used routinely in applications ranging from revenue management and medicine to bioinformatics. **ODTlearn** is a python package for learning various types of decision trees including:

- **StrongTrees** for learning optimal classification trees.
- **FairTrees** for learning optimal classification trees that can incorporate various notions of fairness such as statistical parity, conditional statistical parity, predictive equality, equal opportunity and equalized odds.
- **RobustTrees** for learning optimal classification trees that are robust against distribution shift in the training data.
- **PrescriptiveTrees** for learning optimal prescriptive trees which is a tree for prescription rather than classification where in the leaf nodes it prescribes a treatment.

## Resources for getting started

There are a few ways to get started with ODTlearn:

* Read the [Installation Guide](installation).
* Read the [introductory tutorials](auto_examples/index) for each of the methods implemented in ODTlearn.

## Documentation structure

We provide an overview of the structure of our documentation to help you know where to look when you run into any issues.

* **Tutorials** walk through fitting decision trees for several toy problems using ODTlearn. Start here if you are new to ODTlearn, or you have a particular type of problem you want to model.
* The **API Reference** contains a complete list of the classes and methods you can use in ODTlearn. Go here to know how to use a particular classifier and its corresponding methods.
* The **Developer docs** section contains information for people interested in contributing to ODTlearn development or writing an ODTlearn extension. Don't worry about this section if you are using ODTlearn to solve problems as a user.

## Usage Example

The following script demonstrates how to use the ODTlearn package to fit a StrongTree. For examples of how to use other types of trees please consult the [tutorials](auto_examples/index)
and [API documentation](api).

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from odtlearn.StrongTree import StrongTreeClassifier

data = pd.read_csv("./data/balance-scale_enc.csv")
y = data.pop("target")

X_train, X_test, y_train, y_test = train_test_split(
data, y, test_size=0.33, random_state=42
)

stcl = StrongTreeClassifier(
    depth=1,
    time_limit=60,
    _lambda=0,
    benders_oct=False,
    num_threads=None,
    obj_mode="acc",
)

stcl.fit(X_train, y_train, verbose=True)
stcl.print_tree()
test_pred = stcl.predict(X_test)
print(
    "The out-of-sample acc is {}".format(np.sum(test_pred == y_test) / y_test.shape[0])
)
```

## References
* Aghaei, S., Gómez, A., & Vayanos, P. (2021). Strong optimal classification trees. arXiv preprint arXiv:2103.15965. [\[arxiv\]](https://arxiv.org/abs/2103.15965)
* Jo, N., Aghaei, S., Benson, J., Gómez, A., & Vayanos, P. (2022). Learning optimal fair classification trees. arXiv preprint arXiv:2201.09932. [\[arxiv\]](https://arxiv.org/abs/2201.09932)
* Justin, N., Aghaei, S., Gomez, A., & Vayanos, P. (2021). Optimal Robust Classification Trees. In The AAAI-22 Workshop on Adversarial Machine Learning and Beyond. [\[link\]](https://openreview.net/pdf?id=HbasA9ysA3)
* Jo, N., Aghaei, S., Gómez, A., & Vayanos, P. (2021). Learning optimal prescriptive trees from observational data. arXiv preprint arXiv:2108.13628. [\[arxiv\]](https://arxiv.org/pdf/2108.13628.pdf)


```{toctree}
:caption: Getting Started
:hidden: true
:maxdepth: 2

installation
<!-- quick_start -->
```

```{toctree}
:caption: API Reference
:hidden: true
:maxdepth: 2

api
```

```{toctree}
:caption: Tutorials
:hidden: true
:maxdepth: 1

notebooks/StrongTree
notebooks/FairTree
notebooks/RobustTree
notebooks/PrescriptiveTree
```

```{toctree}
:caption: Developer Docs
:hidden: true
:maxdepth: 1

contributing
```

