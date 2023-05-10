Simulation Setup
################

Simulation setup defines the run-time behaviour of the simulation

Parameter definitions
=====================

**Parameter Name** : ``quadrature``
-----------------------------------

**default value** :

.. code:: yaml

    quadrature:
        quadrature-type: GLL4

**possible values**: [YAML Node]

**documentation** : quadrature parameter defines the type of quadrature used for the simulation (X and Z dimension).

**Parameter Name** : ``quadrature.alpha``
******************************************

**default value** : None

**possible values** : [float, double]

**documentation** : Alpha value of the Gauss-Jacobi quadrature. For GLL quadrature alpha = 0.0

**Parameter Name** : ``quadrature.beta``
*****************************************

**default value** : None

**possible values** : [float, double]

**documentation** : Beta value of the Gauss-Jacobi quadrature. For GLL quadrature beta = 0.0, and for GLJ quadrature beta = 1.0

**Parameter Name** : ``quadrature.ngllx``
******************************************

**default value** : None

**possible values** : [int]

**documentation** : Number of GLL points in X-dimension

**Parameter Name** : ``quadrature.ngllz``
******************************************

**default value** : None

**possible values** : [int]

**documentation** : Number of GLL points in Z-dimension

.. note::

    Below is example of 4th order GLL quadrature with 5 GLL points:

    .. code:: yaml

        quadrature:
            alpha: 0.0
            beta: 0.0
            ngllx: 5
            ngllz: 5

**Parameter Name** : ``quadrature.quadrature-type``
***************************************************

**default value** : None

**possible values** : [GLL4, GLL7]

**decumentation** : Predefined quadrature types.

1. ``GLL4`` defines 4th order GLL quadrature with 5 GLL points.
2. ``GLL7`` defines 7th order GLL quadrature with 8 GLL points.

.. note::

    4th order GLL quadrature with 5 GLL points using pre-defined quadrature types:

    .. code:: yaml

        quadrature:
            quadrature-type: GLL4

**Parameter Name** : ``solver``
-------------------------------

**default value** : None

**possible values** : [YAML Node]

**documentation** : Type of solver to use for the simulation.

**Parameter Name** : ``solver.time-marching``
**********************************************

**default value** : None

**possible values** : [YAML Node]

**documentation** : Select either a time-marching or an explicit solver. Only time-marching solver is implemented currently.

**Parameter Name** : ``solver.time-marching.type-of-simulation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**default value** : forward

**possible values** : [forward]

**documentation** : Select the type of simulation. Forward, backward or adjoint.

**Parameter Name** : ``solver.time-marching.time-scheme``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**default value** : None

**possible values** : [YAML Node]

**documentation** : Select the time-marching scheme.

**Parameter Name** : ``solver.time-marching.time-scheme.type``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**default value** : Newmark

**possible values** : [Newmark]

**documentation** : Select time scheme for the solver

**Parameter Name** : ``solver.time-marching.time-scheme.dt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**default value** : None

**possible values** : [float, double]

**documentation** : Value of time step in seconds

**Parameter Name** : ``solver.time-marching.time-scheme.nstep``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**default value** : None

**possible values** : [int]

**documentation** : Total number of time steps in the simulation
