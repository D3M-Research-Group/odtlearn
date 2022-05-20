% odtlearn documentation master file

# Welcome to odtlearn's documentation!

This site contains the documentation for the odtlearn package.

## What is ODTlearn?


Decision trees are among the most popular and inherently interpretable machine learning models and are used routinely in applications ranging from revenue management and medicine to bioinformatics. **ODTlearn** is a python package for learning various types of decision trees including:

- **StrongTrees** for learning optimal classification trees
- **FairTrees** for learning optimal classification trees that can incorporate various notions of fairness such as statistical parity, conditional statistical parity, predictive equality, equal opportunity and equalized odds
- **RobustTrees** for learning optimal classification trees that are robust against distribution shift in the training data
- **PrescriptiveTrees** for learning optimal prescriptive trees which is a tree for prescription rather than classification where in the leaf nodes it prescribes a treatment 

```{toctree}
:caption: Getting Started
:hidden: true
:maxdepth: 2

quick_start
```

```{toctree}
:caption: Documentation
:hidden: true
:maxdepth: 2

api
```

```{toctree}
:caption: Tutorial - Examples
:hidden: true
:maxdepth: 1

notebooks/StrongTree
notebooks/FairTree
```

## [Getting started](quick_start)

Information how to install and use this package.

## [API Documentation](api)

API documentation for the package.

## [Examples](auto_examples/index)

A set of examples.
