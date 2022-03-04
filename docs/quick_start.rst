#####################################
Getting started with the Trees Package
#####################################

This package contains implementations of several decision tree algorithms including StrongTrees, FairTrees, RobustTrees, and Prescriptive Trees.

Install the package
===================

To install the package, you need to run the following command in your terminal:

.. code-block:: console

   pip install git+https://github.com/patrickvossler18/decision_tree_estimators.git#egg=trees


Usage Example
=============

The following script demonstrates how to use the trees package to fit a StrongTree. For examples of how to use other types of trees please consult the :ref:`user guide <user_guide>`
and :ref:`API documentation <api>`.

.. code-block:: python

    from trees.StrongTree import StrongTreeClassifier

