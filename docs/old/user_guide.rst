.. title:: User guide : contents

.. _user_guide:

.. _`Aghaei et al., 2021`: https://arxiv.org/abs/2103.15965

.. |StrongTree_graph| image:: _static/img/StrongTree_graph.png
  :width: 400
  :alt: Diagram showing the conversion of a tree of fixed depth to a directed acyclic graph with sink and source nodes

==================================================
User guide: fitting ODTlearn estimators
==================================================

Introduction to StrongTrees
===========================

This document aims to show how to use the `StrongTreeClassifier`. We begin with the standard use case, walking through parameter choices and method details, and then provide a small example on a real-world data set.


StrongTreeClassifier: the basics
--------------------------------

StrongTree is an MIO formulation for learning optimal *balanced* classification tress of a given depth, i.e., trees wherein the distance between all nodes where a prediction is made, and the root node is equal to the tree depth. This MIO formulation relies on the observation that once the structure of the tree is fixed, determining whether a datapoint is correctly classified or not reduces to checking whether the datapoint can, based on its feature and label, flow from the root of the tree to a leaf where the prediction made matches its label. For a complete treatment of StrongTrees, see our paper `Aghaei et al., 2021`_.

A key step towards our flow-based MIO formulation of the problem consists of converting the decision tree of fixed depth that we wish to train to a directed acyclic graph where all arcs are directed from the root of the tree to the leaves:

|StrongTree_graph|


This modification of the tree enables us to think of the decision tree as a *directed acyclic graph with a single source and sink node*. Datapoints *flow* from the source to sink through a single path and only reach the sink if they are correctly classified.


.. abbreviated version of problem formulation here

Toy example
-----------
We start with a simple example and investigate different parameter combinations to provide intuition on how they affect the structure of the tree.

First we generate the data for our example. The diagram within the code block shows the expected structure of the fitted decision tree.
.. code-block:: python

    import numpy as np
    '''
        X2
        |               |
        |               |
        1    + +        |    -
        |               |   
        |---------------|-------------
        |               |
        0    - - - -    |    + + +
        |    - - -      |
        |______0________|_______1_______X1
    '''
    X = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                [1,0],[1,0],[1,0],
                [1,1],
                [0,1],[0,1]])

    y = np.array([0,0,0,0,0,0,0,
                1,1,1,
                0,
                1,1])

.. code-block:: python
    from odtlearn.StrongTree import StrongTreeClassifier
    import numpy as np
 
    stcl = StrongTreeClassifier(
            depth = 2, 
            time_limit = 60,
            _lambda = 0.51,
            benders_oct= True, 
            num_threads=None, 
            obj_mode = 'acc'
        )
    stcl.fit(X, y, verbose = False)
    stcl.print_tree()
    predictions = stcl.predict(X)
    print(f'\n\n In-sample accuracy is {np.sum(predictions==y)/y.shape[0]}')


FairTree
--------

RobustTree
----------

PrescriptiveTree
----------------

