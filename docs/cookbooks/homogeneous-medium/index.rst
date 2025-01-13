.. _homogeneous_example:

Wave propagration through homogeneous media
===========================================

In this `example <https://github.com/PrincetonUniversity/SPECFEMPP/tree/main/examples/homogeneous-medium-flat-topography>`_ we simulate wave propagation through a 2-dimensional homogeneous medium.

Setting up your workspace
--------------------------

Let's start by creating a workspace from where we can run this example.

.. code-block:: bash

    mkdir -p ~/specfempp-examples/homogeneous-medium-flat-topography
    cd ~/specfempp-examples/homogeneous-medium-flat-topography

We also need to check that the SPECFEM++ build directory is added to the ``PATH``.

.. code:: bash

    which specfem2d

If the above command returns a path to the ``specfem2d`` executable, then the build directory is added to the ``PATH``. If not, you need to add the build directory to the ``PATH`` using the following command.

.. code:: bash

    export PATH=$PATH:<PATH TO SPECFEM++ BUILD DIRECTORY/bin>

.. note::

    Make sure to replace ``<PATH TO SPECFEM++ BUILD DIRECTORY/bin>`` with the actual path to the SPECFEM++ build directory on your system.

Now let's create the necessary directories to store the input files and output artifacts.

.. code:: bash

    mkdir -p OUTPUT_FILES
    mkdir -p OUTPUT_FILES/seismograms

    touch specfem_config.yaml
    touch single_source.yaml
    touch topography_file.dat
    touch Par_File

Generating a mesh
-----------------

To generate the mesh for the homogeneous media we need a parameter file, ``Par_File``, a topography file, `topography_file.dat`, and the mesher executible, ``xmeshfem2D``, which should have been compiled during the installation process.

.. note::
  Currently, we still use a mesher that was developed for the original `SPECFEM2D <https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/>`_ code. More details on the meshing process can be found `here <https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/>`_.

We first define the meshing parameters in a Parameter file.

Parameter File
~~~~~~~~~~~~~~~~

.. literalinclude:: Par_file
    :caption: Par_file
    :language: bash
    :emphasize-lines: 10-11,123-124,
    :linenos:


At this point, it is worthwhile to note few key parameters within the
``PAR_FILE`` as it pertains to SPECFEM++.

- This version of SPECFEM++ does not support simulations running across multiple
  nodes, i.e., we have not enabled MPI. Relevant parameter value:

.. code:: bash

        NPROC   = 1

- The path to the topography file is provided using the ``interfacesfile``
  parameter. Relevant values:

.. code:: bash

    interfacesfile = topography_file.dat

.. _homogeneous-medium-flat-topography-topography-file:

Topography file
~~~~~~~~~~~~~~~~~

.. code-block:: bash
    :caption: topography_file.dat
    :linenos:

    #
    # number of interfaces
    #
     2
    #
    # for each interface below, we give the number of points and then x,z for each point
    #
    #
    # interface number 1 (bottom of the mesh)
    #
     2
     0 0
     5000 0
    # interface number 2 (topography, top of the mesh)
    #
     2
        0 3000
     5000 3000
    #
    # for each layer, we give the number of spectral elements in the vertical direction
    #
    #
    # layer number 1 (bottom layer)
    #
     60

Running ``xmeshfem2D``
~~~~~~~~~~~~~~~~~~~~~~

To execute the mesher run

.. code:: bash

    xmeshfem2D -p Par_File

Check the mesher generated files in the ``OUTPUT_FILES`` directory.

.. code:: bash

    ls -ltr OUTPUT_FILES

Defining sources
----------------

Next we define the sources using a YAML file. For full description on parameters used to define sources refer :ref:`source_description`.

.. code-block:: yaml
    :linenos:
    :caption: single_source.yaml

    number-of-sources: 1
    sources:
      - force:
          x : 2500.0
          z : 2500.0
          source_surf: false
          angle : 0.0
          vx : 0.0
          vz : 0.0
          Ricker:
            factor: 1e10
            tshift: 0.0
            f0: 10.0

Configuring the solver
-----------------------

Now that we have generated a mesh and defined the sources, we need to set up the solver. To do this we define another YAML file ``specfem_config.yaml``. For full description on parameters used to define sources refer :ref:`parameter_documentation`.

.. code-block:: yaml
    :linenos:
    :caption: specfem_config.yaml

    parameters:

      header:
        ## Header information is used for logging. It is good practice to give your simulations explicit names
        title: Isotropic Elastic simulation # name for your simulation
        # A detailed description for your simulation
        description: |
          Material systems : Elastic domain (1)
          Interfaces : None
          Sources : Force source (1)
          Boundary conditions : Neumann BCs on all edges

      simulation-setup:
        ## quadrature setup
        quadrature:
          quadrature-type: GLL4

        ## Solver setup
        solver:
          time-marching:
            time-scheme:
              type: Newmark
              dt: 1.1e-3
              nstep: 1600

        simulation-mode:
          forward:
            writer:
              seismogram:
                format: "ascii"
                directory: OUTPUT_FILES/seismograms

      receivers:
        stations-file: OUTPUT_FILES/STATIONS
        angle: 0.0
        seismogram-type:
          - velocity
        nstep_between_samples: 1

      ## Runtime setup
      run-setup:
        number-of-processors: 1
        number-of-runs: 1

      ## databases
      databases:
        mesh-database: OUTPUT_FILES/database.bin
        source-file: single_source.yaml

At this point lets focus on a few sections in this file:

- Configure the solver using ``simulation-setup`` section.

.. code-block:: yaml

    simulation-setup:
      ## quadrature setup
      quadrature:
        quadrature-type: GLL4
      ## Solver setup
      solver:
        time-marching:
          time-scheme:
            type: Newmark
            dt: 1.1e-3
            nstep: 1600
      simulation-mode:
        forward:
          writer:
            seismogram:
              format: "ascii"
              directory: OUTPUT_FILES/seismograms

* We first define the integration quadrature to be used in the simulation. At this moment, the code supports a 4th order Gauss-Lobatto-Legendre quadrature with 5 GLL points (``GLL4``) & a 7th order Gauss-Lobatto-Legendre quadrature with 8 GLL points (``GLL7``).
* Define the solver scheme using the ``time-scheme`` parameter.
* Define the simulation mode to be forward and the output format for synthetic seismograms seismograms.

- Define the path to the meshfem generated database file using the ``mesh-database`` parameter and the path to source description file using ``source-file`` parameter. Relevant parameter values:

.. code-block:: yaml

    ## databases
    databases:
      mesh-database: OUTPUT_FILES/database.bin
      source-file: single_source.yaml

- It is good practice to have distinct header section for you simulation. These sections will be printed to standard output during runtime helping the you to distinguish between runs using standard strings. Relevant paramter values

.. code-block:: yaml

    header:
      ## Header information is used for logging. It is good practice to give your simulations explicit names
      title: Isotropic Elastic simulation # name for your simulation
      # A detailed description for your simulation
      description: |
        Material systems : Elastic domain (1)
        Interfaces : None
        Sources : Force source (1)
        Boundary conditions : Neumann BCs on all edges

Running the solver
-------------------

Finally, to run the SPECFEM++ solver

.. code:: bash

    specfem2d -p specfem_config.yaml

.. note::

    Make sure either your are in the build directory of SPECFEM++ or the build directory is added to your ``PATH``.

Visualizing seimograms
----------------------

Let us now plot the traces generated by the solver using ``obspy``. This version of the code only supports ASCII output format for seismograms. To plot the seismograms we need to read the ASCII files as ``numpy`` arrays and them convert them to ``obspy`` streams. The following code snippet shows how to do this.

.. code-block:: python

    import os
    import numpy as np
    import obspy

    def get_traces(directory):
        traces = []
        ## iterate over all seismograms
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            station_name = os.path.splitext(filename)[0]
            trace = np.loadtxt(f, delimiter=' ')
            starttime = trace[0,0]
            dt = trace[1,0] - trace[0,0]
            traces.append(obspy.Trace(trace[:,1], {'network': station_name, 'starttime': starttime, 'delta': dt}))

        stream = obspy.Stream(traces)

        return stream

    directory = OUTPUT_FILES/seismograms
    stream = get_traces(directory)
    stream.plot(size=(800, 1000))

.. figure:: ../../examples/homogeneous-medium-flat-topography/traces.png
   :alt: Traces
   :width: 800
   :align: center

   Traces.
