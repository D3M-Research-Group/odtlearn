# Robust Optimal Classification Trees
In many applications, full control of data collection in both training and deployment is rare. For example, the data collection mechanism may change between training and deployment, or the environment may change the distribution of data over time. This corresponds to a *distribution shift*, where the distribution of the training data does not match the distribution of the deployment data. As a result, any trained model can perform poorly in the testing/deployment phase in the presence of distribution shifts

`RobustTreeClassifier` is an MIO-based method for building optimal classification trees robust to these distribution shifts for data with integer-valued features. Details on the method can be found in the paper (Justin et al. 2021).

<p align="center">
    <img src="../_static/img/distribution-shift.png" alt="robust_shift" style="width:800px;"/>
</p> 

## Specifying the Distribution Shift
To fit a `RobustTreeClassifier`, the expected distribution shift must be specified. This is through the `costs` and `budget` parameters of the `RobustTreeClassifier.fit()` function. `RobustTreeClassifier` contains a function `probabilities_to_costs` to help generate the values for `costs` and `budget` based on knowledge from the application.

The `prob` parameter of `probabilities_to_costs` is a matrix (of the same shape as the training covariates) where each entry contains the estimated probability that feature $f$ of sample $i$ will not be shifted, which can be set based on domain knowledge.

The `threshold` parameter of `probabilities_to_costs` tunes how much robustness to distribution shifts is needed (in exchange for solving time), and is some value from 0 (exclusive) to 1 (inclusive), where 1 represents no robustness to uncertainty and values near 0 represent complete robustness to uncertainty (i.e. a tree that does not branch). In most settings, a reasonable range for this parameter is between 0.7 and 1. It is advised to tune this parameter and to try different values.

For details on how the costs and budgets are mathematically derived from specified `prob` and `threshold` values, see the paper (Justin et al. 2021).

<!--
<p align="left">
    <img src="../_static/img/robust_uncertainty.png" alt="robust_uncertainty" style="width:400px;"/>
</p> 
-->

## References
* Justin, N., Aghaei, S., Gómez, A., & Vayanos, P. (2021). Optimal robust classification trees. *The AAAI-2022 Workshop on Adversarial Machine Learning and Beyond*. <https://openreview.net/pdf?id=HbasA9ysA3>