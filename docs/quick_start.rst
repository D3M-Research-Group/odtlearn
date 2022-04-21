#########################################
Getting started with the ODTlearn Package
#########################################

This package contains implementations of several decision tree algorithms including StrongTrees, FairTrees, RobustTrees, and Prescriptive Trees.

Install the package
===================

To install the package, you need to run the following command in your terminal:

.. code-block:: bash

    pip install git+https://github.com/D3M-Research-Group/odtlearn.git#egg=odtlearn


Usage Example
=============

The following script demonstrates how to use the odtlearn package to fit a StrongTree. For examples of how to use other types of trees please consult the :ref:`examples <auto_examples/index.html>`
and :ref:`API documentation <api>`.

.. code-block:: python

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


References
==========
* Aghaei, S., Gómez, A., & Vayanos, P. (2021). Strong optimal classification trees. arXiv preprint arXiv:2103.15965. `[arxiv] <https://arxiv.org/abs/2103.15965>`_ 
* Jo, N., Aghaei, S., Benson, J., Gómez, A., & Vayanos, P. (2022). Learning optimal fair classification trees. arXiv preprint arXiv:2201.09932. `[arxiv] <https://arxiv.org/abs/2201.09932>`_
* Justin, N., Aghaei, S., Gomez, A., & Vayanos, P. (2021). Optimal Robust Classification Trees. In The AAAI-22 Workshop on Adversarial Machine Learning and Beyond. `[link] <https://openreview.net/pdf?id=HbasA9ysA3>`_
* Jo, N., Aghaei, S., Gómez, A., & Vayanos, P. (2021). Learning optimal prescriptive trees from observational data. arXiv preprint arXiv:2108.13628. `[arxiv] <https://arxiv.org/pdf/2108.13628.pdf>`_ 