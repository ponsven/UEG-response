.. UEG-response documentation master file, created by
   sphinx-quickstart on Mon Sep 21 14:07:43 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

UEG-response documentation
==========================

.. note:: 
   This packgae is under active development. The API may still change.


Welcome to the documentation for *UEG-response*! 
Below the installation steps and the packgae API is given.


Installation
------------

To use *UEG-response*, first install the package locally:

* [Optional] Activate the virtual enviroment where the *UEG-response* should be installed.
* [Initial build] For the initial build you might need to install `build`:

.. code-block:: console

    python -m pip install --upgrade build

* Build and install *UEG-response*:

.. code-block:: console

    python -m build
    python -m pip install -e .



Evaluation
----------

A basic use case of the package is to evaluate the quadratic response function. 
This could looks like this:

.. code-block:: python

    # Import packages
    import numpy as np
    import UEG_response as ur

    # Define basic properties
    # Condition
    rs = 3.23
    theta = 1.0
    inv_theta = 1/theta

    # Units
    hbar = 1.0
    aB = 1.0
    m = 1.0
    e = 1.0
    eps0 = 1/(4*np.pi)

    # Normalisation
    qF = (9*np.pi/4)**(1/3) / (rs*aB)
    EF = hbar**2 * qF**2 / (2*m)
    beta = 1/(theta*EF)
    n = 3/(4*np.pi*rs**3)

    # Input 
    omega1 = np.linspace(-5.0, 5.0, 100) * EF/hbar
    omega2 = 0.0
    k1 = 0.5 * qF
    k2 = 1.0 * qF
    csTheta = 0.7

    ur.ideal_quadratic_response(omega1, k1, omega2, k2, csTheta, m, hbar, n, beta)    


Additional exampels are provided in the 'exampels' folder.

Further information
-------------------

Check out the :doc:`api` section for further information on top level functionality.

.. toctree::
   api
   :maxdepth: 2
   :caption: Contents:

