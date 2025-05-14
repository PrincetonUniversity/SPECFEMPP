
Velocity Models
================

Velocity models are defined in the ``meshfem2d`` module. The velocity model is
defined in the ``model`` section of the input file. The velocity model is
defined by a set of parameters that describe the material properties of the
model. An example of a velocity model is shown below:

.. code-block:: bash

    # number of material systems
    nbmodels = 2
    # acoustic elastic material system
    1 1 2500.d0 3400.d0 1963.d0 0 0 9999 9999 0 0 0 0 0 0
    2 1 1020.d0 1500.d0 0.d0 0 0 9999 9999 0 0 0 0 0 0

Note that ``nbmodels`` must always be followed by the number of material systems
in the model, one for each material system. See paramter descriptions below for
more details on the parameters.

Metaparameters
--------------


``nbmodels``
~~~~~~~~~~~~

Number of material systems in the model.

:Type: ``int``

.. code-block::
    :caption: Example

    nbmodels = 1


``TOPOGRAPHY_FILE``
~~~~~~~~~~~~~~~~~~~

Path to an external topography file.

:Type: ``string``

.. code-block::
    :caption: Example

    TOPOGRAPHY_FILE = topography.dat


``read_external_mesh``
~~~~~~~~~~~~~~~~~~~~~~

If ``True`` the mesh is read from an external file.

Type
    ``logical``

.. code-block::
    :caption: Example

    read_external_mesh = .true.


Description of material system
------------------------------

Each material system in the model is described by a string.

.. note::
    Only elastic, poroelastic, and acoustic material systems are supported.


Elastic material system
~~~~~~~~~~~~~~~~~~~~~~~

An elastic medium can be described by the following parameters:

- ``model_number``: integer number to refence the material system
- ``rho``: density
- ``Vp``: P-wave velocity
- ``QKappa``: Attenuation parameter (set to ``9999`` for no attenuation)

:Type: ``string``

:Format: ``model_number 1 rho Vp 0 0 QKappa 9999 0 0 0 0 0 0``

.. code-block::
    :caption: Example

    1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0

Acoustic material system
~~~~~~~~~~~~~~~~~~~~~~~~

An acoustic medium can be described by the following parameters:

- ``model_number``: integer number to refence the material system
- ``rho``: density
- ``Vp``: P-wave velocity
- ``Vs``: S-wave velocity
- ``QKappa``: Attenuation parameter (set to ``9999`` for no attenuation)
- ``QMu``: Attenuation parameter (set to ``9999`` for no attenuation)

Type
    ``string``

:Format: ``model_number 1 rho Vp Vs 0 0 QKappa QMu 0 0 0 0 0 0``

.. code-block::
    :caption: Example

    1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0


Poroelastic material system
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A poroelastic medium can be described by the following parameters:

- ``model_number``: integer number to refence the material system
- ``rhos``: solid density
- ``rhof``: fluid density
- ``phi``: porosity
- ``c``: Biot coefficient
- ``kxx``: permeability in x direction
- ``kxz``: permeability in z direction
- ``kzz``: permeability in z direction
- ``Ks``: bulk modulus of solid
- ``Kf``: bulk modulus of fluid
- ``Kfr``: bulk modulus of fluid in the frame
- ``etaf``: viscosity of fluid
- ``mufr``: shear modulus of fluid in the frame
- ``Qmu``: attenuation parameter (set to ``9999`` for no attenuation)

:Type: ``string``

:Format: ``model_number 3 rhos rhof phi c kxx kxz kzz Ks Kf Kfr etaf mufr Qmu``

.. code-block::
    :caption: Example

    1 3 2650.d0 880.d0 0.1d0 2.0 1d-9 0.d0 1d-9 12.2d9 1.985d9 9.6d9 0.d0 5.1d9 9999
