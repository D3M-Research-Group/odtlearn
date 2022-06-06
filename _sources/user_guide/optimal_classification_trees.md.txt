# Optimal Classification Trees

This document shows how to use the `StrongTreeClassifier` to train optimal classification trees. It introduces StrongTree, demonstrates the standard use case, walks through parameter choices and method details, and then provides a small example using a dataset from UCI repository.


## The Basics

Classification trees are among the most popular and inherently interpretable machine learning models and are used routinely in applications ranging from revenue management and medicine to bioinformatics. In these settings, it is crucial to leverage the full potential of available data to obtain the best possible, a.k.a, optimal, classification tree. **StrongTree** is a Mixed-Integer-Optimization (MIO) based formulation for learning optimal classification trees of a given depth. StrongTree significantly improves upon traditional heuristic approaches in terms of performance and is much faster than other MIO-based alternatives. See the corresponding paper for a complete treatment of StrongTree (Aghaei et al., 2021). 

### Handling Integer Features

StrongTree requires all the input features to be binary, i.e., all the elements of feature matrix $X$ should be either 0 or 1. However, this formulation can also be applied to datasets involving categorical or integer features by first preprocessing the data. For each categorical feature, one needs to encode it as a one-hot vector, i.e., for each level of the feature, one needs to create a new binary column with a value of one if and only if the original column has the corresponding level. A similar approach for encoding integer features can be used with a slight change. The new binary column should have a value of one if and only if the main column has the corresponding value or any value smaller than it. 
ODTlearn provides a function called `binarize` for taking care of the binarization step where given a dataframe with only categorical/integer columns it outputs a binarized dataframe.



### Imbalanced Decision Trees

To avoid overfitting on the training data, `StrongTree` can add a regularization term (with the regularization parameter $0 \leq \lambda \leq 1$) to the objective function. This regularization term penalizes the number of branching nodes in the decision tree. Higher values of $\lambda$ results in a sparser decision tree, i.e., a decision tree where the distance of its leaf nodes from the root node is not the same. Figure 2 shows an example of an imbalanced decision tree. You can set the value of $\lambda$ via parameter `_lambda` in `StrongTreeClassifier`.


```{image} ../_static/img/classification_tree_imbalanced.png
:alt: An imbalanced decision tree of depth 2
:width: 600px
:align: center
```

<!-- <p align = "center">
Figure.2 - An imbalanced decision tree of depth 2
</p> -->

## Different Choice of Objective

`StrongTreeClassifier` provides two options for the objective function. It can either maximize the classification accuracy or the balanced accuracy, which averages the accuracy across classes.  

The balanced accuracy can be helpful in the case of imbalanced datasets. A dataset is imbalanced when the class distribution is not uniform, i.e., when the number of data points in each class varies significantly between classes. For an imbalanced dataset, predicting the majority class results in high accuracy. Thus decision trees that maximize prediction accuracy without accounting for the imbalanced nature of the data perform poorly on the minority class. Imbalanced datasets occur in many important domains, e.g., in a problem such as suicide prevention, there are very small number of suicide instances.


## Optimality

To use ODTlearn, you need to have the **Gurobi** solver installed on your machine. Depending on the size of the problem (number of data points and features), the `StrongTreeClassifier` needs a certain amount of time to find the optimal decision tree. For larger problems, this solving time may be hours long. For this reason, you need to provide a time limit, and the solving process times out after the given time limit. The default time limit for the `StrongTreeClassifier` is 60 seconds (`time_limit=60`). Still, you may need to increase the time limit for larger problems if the classifier cannot find a high-quality solution. `StrongTreeClassifier` either finds the optimal tree within the time limit or reports the optimality gap, which is an indicator of the quality of the solution, i.e., how far our solution is from the optimal solution (in the case of optimality, the gap is zero).



## Benders' Decomposition

ODTlearn offers a novel Benders' decomposition-based algorithm for solving the problem of learning optimal classification trees. This algorithm is the default algorithm for `StrongTreeClassifier`. However, one may not want to use the decomposition algorithm, i.e., `benders_oct= False`, wherein, in this case, we directly solve the MIO formulation using Gurobi. The Benders' decomposition-based algorithm is significantly faster than the direct approach.
