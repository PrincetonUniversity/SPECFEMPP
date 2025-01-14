Wave propagration through solid-solid interface
===============================================

In this cookbook we are simulating a medium with two solids. This is a classic
example from the computational seismology class at Princeton University, in which
students are familiarized with wave propagation through an elastic medium.

The model that we are using is a 2D model with a solid-solid interface. The
characteristics of the medium are shown in the figure below.

.. figure:: model.png
    :width: 600
    :alt: solid-solid-interface

    Solid-solid interface model with a slow material on top and a fast material
    at the bottom.

The model consists of two materials, one with a slower velocity and the other
with a faster velocity. The model is divided by a horizontal interface. The
source and the receiver are both indicated in the figure, and located at the
surface of the model.

Setting up the workspace
------------------------

Let's start by creating a workspace from where we can run this example.

.. code-block:: bash

    mkdir -p ~/specfempp-examples/solid-solid-interface
    cd ~/specfempp-examples/solid-solid-interface

We also need to check that the SPECFEM++ build directory is added to the ``PATH``.

.. code:: bash

    which specfem2d

If the above command returns a path to the ``specfem2d`` executable, then the
build directory is added to the ``PATH``. If not, you need to add the build
directory to the ``PATH`` using the following command.

.. code:: bash

    export PATH=$PATH:<PATH TO SPECFEM++ BUILD DIRECTORY/bin>

.. note::

    Make sure to replace ``<PATH TO SPECFEM++ BUILD DIRECTORY/bin>`` with the
    actual path to the SPECFEM++ build directory on your system.

Now let's create the necessary directories to store the input files and output
artifacts.

.. code:: bash

    mkdir -p OUTPUT_FILES
    mkdir -p OUTPUT_FILES/seismograms

    touch specfem_config.yaml
    touch sources.yaml
    touch topography.dat
    touch Par_File


Meshing the domain
------------------

We first start by generating a mesh for our simulation domain using ``xmeshfem2D``. To do this, we first define our simulation domain and the meshing parmeters in a parameter file.

Parameter file
~~~~~~~~~~~~~

.. literalinclude:: Par_file
    :caption: Par_file
    :language: bash
    :linenos:
    :emphasize-lines: 58-78


- Like we did in the :ref:`homogeneous_example`, we define the elastic velocity
  model layers in the `Velocity and density models` section of the parameter
  file. This time, however, we define two material systems with different
  elastic parameters as defined in the figure above. First, we adjust the number
  of model materials to 2 using the ``nbmodels`` parameter.

  .. literalinclude:: Par_file
      :caption: Par_file
      :start-at: nbmodels
      :end-at: nbmodels
      :lineno-match:
      :linenos:
      :language: bash

  Then, we then define the velocity model for each material based on the
  parameters in the figure above. We define the elastic material

  We use the conversion from :math:`\kappa`, :math:`\mu` and :math:`\rho`
  to :math:`v_p` and :math:`v_s`:

  .. math::

      v_p = \sqrt{\frac{kappa + 4/3 \mu}{rho}}
      v_s = \sqrt{\frac{\mu}{rho}}

  and add both materials using the format:

  .. code-block::bash

      model_number rho Vp Vs 0 0 QKappa Qmu 0 0 0 0 0 0``.

  .. literalinclude:: Par_file
      :caption: Par_file
      :start-at: 1 1
      :end-at: 2 1
      :lineno-match:
      :linenos:

  As you can see, we added two material systems in the simulation domain. With
  the properties used from the model where the faster material is model number
  1 and the slower material is model number 2.

- Additionally, we define stacey absorbing boundary conditions on all the edges
  of the domain except the top surface using the ``STACEY_ABSORBING_BOUNDARY``,
  ``absorbbottom``, ``absorbright``, ``absorbtop`` and ``absorbleft``
  parameters.

  .. literalinclude:: Par_file
      :caption: Par_file
      :start-at: STACEY_ABSORBING_BOUNDARY
      :end-at: absorbleft
      :lineno-match:
      :linenos:

Defining the topography of the domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define the bounds and topography of the domain using the following topography
file

.. literalinclude:: topography_file.dat
    :caption: topography_file.dat
    :language: bash
    :linenos:
    :emphasize-lines: 11-13,17-19,23-25

With 38 elements vertically in each layer.


Running ``xmeshfem2D``
~~~~~~~~~~~~~~~~~~~~~~

To execute the mesher run

.. code:: bash

    xmeshfem2D -p Par_File

.. note::

    Make sure either your are in the build directory of SPECFEM2D kokkos or the build directory is added to your ``PATH``.

Note the path of the database file and a stations file generated after successfully running the mesher.

Defining the source
~~~~~~~~~~~~~~~~~~~

We define the source location and the source time function in the source file.

.. code-block:: yaml
    :caption: single_source.yaml

    number-of-sources: 1
    sources:
      - force:
          x : 1575.0
          z : 2900.0
          source_surf: false
          angle : 0.0
          vx : 0.0
          vz : 0.0
          Ricker:
            factor: 1e9
            tshift: 0.0
            f0: 10.0

Running the simulation
----------------------

To run the solver, we first need to define a configuration file ``specfem_config.yaml``.

.. code-block:: yaml
    :caption: specfem_config.yaml

    parameters:

      header:
        ## Header information is used for logging. It is good practice to give your simulations explicit names
        title: Heterogeneous acoustic-elastic medium with 1 acoustic-elastic interface (orientation horizontal)  # name for your simulation
        # A detailed description for your simulation
        description: |
          Material systems : Elastic domain (1), Acoustic domain (1)
          Interfaces : Acoustic-elastic interface (1) (orientation horizontal with acoustic domain on top)
          Sources : Force source (1)
          Boundary conditions : Neumann BCs on all edges
          Debugging comments: This tests checks coupling acoustic-elastic interface implementation.
                              The orientation of the interface is horizontal with acoustic domain on top.

      simulation-setup:
        ## quadrature setup
        quadrature:
          quadrature-type: GLL4

        ## Solver setup
        solver:
          time-marching:
            type-of-simulation: forward
            time-scheme:
              type: Newmark
              dt: 0.85e-3
              nstep: 600

        simulation-mode:
          forward:
            writer:
              seismogram:
                format: ascii
                directory: OUTPUT_FILES/seismograms

      receivers:
        stations-file: OUTPUT_FILES/STATIONS
        angle: 0.0
        seismogram-type:
          - displacement
        nstep_between_samples: 1

      ## Runtime setup
      run-setup:
        number-of-processors: 1
        number-of-runs: 1

      ## databases
      databases:
        mesh-database: OUTPUT_FILES/database.bin
        source-file: single_source.yaml

With the configuration file in place, we can run the solver using the following command

.. code:: bash

    specfem2d -p specfem_config.yaml
