######################################
Getting started with the ODTlearn Package
######################################

This package contains implementations of several decision tree algorithms including StrongTrees, FairTrees, RobustTrees, and Prescriptive Trees.

Install the package
===================

To install the package, you need to run the following command in your terminal:

.. code-block:: console

   pip install git+https://github.com/D3M-Research-Group/decision_tree_estimators.git#egg=odtlearn


Usage Example
=============

The following script demonstrates how to use the odtlearn package to fit a StrongTree. For examples of how to use other types of trees please consult the :ref:`examples <auto_examples/index.html>`
and :ref:`API documentation <api>`.

.. code-block:: python

    from odtlearn.StrongTree import StrongTreeClassifier

