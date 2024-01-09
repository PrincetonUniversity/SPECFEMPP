
Velocity Models
================

**Parameter Name**: ``nbmodels``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Number of material systems in the model.

**Type**: ``int``

**Paramter Name**: ``TOPOGRAPHY_FILE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Path to an external topography file.

**Type**: ``string``

**Parameter Name**: ``read_external_mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: If ``True`` the mesh is read from an external file.

**Type**: ``logical``

Description of material system
------------------------------

Each material system in the model is described by a string.

.. note::
    Only elastic and acoustic material systems are supported.

Elastic material system
************************

**string value**: ``model_number 1 rho Vp 0  0 0 QKappa 9999 0 0 0 0 0 0``

**Description**:
    - ``model_number``: integer number to refence the material system
    - ``rho``: density
    - ``Vp``: P-wave velocity
    - ``QKappa``: Attenuation parameter (set to ``9999`` for no attenuation)

**Example**:
    ``1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0``

Acoustic material system
*************************

**string value**: ``model_number 1 rho Vp Vs 0 0 QKappa QMu 0 0 0 0 0 0``

**Description**:
    - ``model_number``: integer number to refence the material system
    - ``rho``: density
    - ``Vp``: P-wave velocity
    - ``Vs``: S-wave velocity
    - ``QKappa``: Attenuation parameter (set to ``9999`` for no attenuation)
    - ``QMu``: Attenuation parameter (set to ``9999`` for no attenuation)

**Example**:
    ``1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0``
