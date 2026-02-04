.. _marmousi_example:

Wave propagation through the Marmousi2 model
============================================

In this example (see :repo-file:`benchmarks/src/dim2/marmousi`) we simulate wave propagation through the 2-dimensional Marmousi2 model,
a complex synthetic velocity model commonly used for testing seismic wave propagation
codes. The model features realistic geological structures with varying velocity gradients,
making it an excellent benchmark for validating numerical methods.

.. note::

    All cookbook files can be copy and pasted from the code blocks below, or you
    can download a zip file containing all the files needed to run this example.

    .. download-folder:: parameter_files
        :filename: marmousi_cookbook.zip
        :text: Marmousi2 cookbook

.. warning::

    **GPU Version Recommended**

    This cookbook example uses the Marmousi2 model, which is computationally intensive.
    It is recommended to run this example only if you have compiled the GPU version of
    SPECFEM++, as it will be too slow on CPUs. Additionally, MPI support is not yet
    implemented in this version.


**Credits**

* **Original Marmousi2 model**: Yann Capdeville (2009)
* **Improved mesh**: Hom Nath Gharti (2016, 2023)
* **CUBIT meshing workflow**: Daniel Peter (Python scripts for streamlined mesh generation)

Setting up your workspace
-------------------------

Let's start by creating a workspace from where we can run this example.

.. code-block:: bash

    mkdir -p ~/specfempp-examples/marmousi
    cd ~/specfempp-examples/marmousi

We also need to check that the SPECFEM++ executable directory is added to the
``PATH``.

.. code:: bash

    which specfem2d

If the above command returns a path to the ``specfem2d`` executable, then the
executable directory is added to the ``PATH``. If not, you need to add the executable
directory to the ``PATH`` using the following command.

.. code:: bash

    export PATH=$PATH:<PATH TO SPECFEM++ DIRECTORY/bin>

.. note::

    Make sure to replace ``<PATH TO SPECFEM++ DIRECTORY/bin>`` with the
    actual path to the SPECFEM++ directory on your system.

Now let's create the necessary directories to store the input files and output
artifacts.

.. code:: bash

    mkdir -p OUTPUT_FILES
    mkdir -p OUTPUT_FILES/results

    touch specfem_config.yaml
    touch sources.yaml
    touch Par_file

Mesh files
----------

Unlike the homogeneous medium example, the Marmousi2 model uses an externally
generated mesh using Coreform Cubit's built in Python interface. The
``MESH-default`` directory contains all the necessary mesh files.

In the following subsections, we plot the first 20 lines of each mesh file used
in this example, along with a brief description of their contents.


.. note::

    The full file content are thousands of lines long, making it impractical to
    include here and render on the webpage. Instead, please download the full
    files using the link provided at the beginning of this cookbook. Or,
    download the just the ``MESH-default`` directory:

    .. download-folder:: parameter_files/MESH-default
        :filename: MESH-default.zip
        :text: MESH-default


mesh_file
~~~~~~~~~

Element connectivity information

.. literalinclude:: parameter_files/MESH-default/mesh_file
   :caption: MESH-default/mesh_file
   :lines: 1-20

nodes_coords_file
~~~~~~~~~~~~~~~~~

Node coordinates

.. literalinclude:: parameter_files/MESH-default/nodes_coords_file
   :caption: MESH-default/nodes_coords_file
   :lines: 1-20

materials_file
~~~~~~~~~~~~~~

Material assignments for each element

.. literalinclude:: parameter_files/MESH-default/materials_file
   :caption: MESH-default/materials_file
   :lines: 1-20

nummaterial_velocity_file_marmousi2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Velocity model properties

.. literalinclude:: parameter_files/MESH-default/nummaterial_velocity_file_marmousi2
   :caption: MESH-default/nummaterial_velocity_file_marmousi2
   :lines: 1-20

free_surface_file
~~~~~~~~~~~~~~~~~

Free surface boundary definition

.. literalinclude:: parameter_files/MESH-default/free_surface_file
   :caption: MESH-default/free_surface_file
   :lines: 1-20

absorbing_surface_file
~~~~~~~~~~~~~~~~~~~~~~

Absorbing boundary conditions

.. literalinclude:: parameter_files/MESH-default/absorbing_surface_file
   :caption: MESH-default/absorbing_surface_file
   :lines: 1-20

Generating the mesh database
----------------------------

To generate the mesh database for SPECFEM++ we need a parameter file (``Par_file``),
the mesh files (in ``MESH-default``), and the mesher executable (``xmeshfem2D``).

.. note::

  This example uses the external mesh capability of SPECFEM2D. The mesh was
  originally created using CUBIT/Trelis, a powerful meshing tool that allows
  for complex geometries and variable mesh refinement.

Parameter File
~~~~~~~~~~~~~~

.. literalinclude:: parameter_files/Par_file
    :caption: Par_file
    :language: bash
    :emphasize-lines: 11,54,64-69,120,130-136

Key parameters for the Marmousi2 model:

- **NPROC**: Set to 1 (MPI not yet supported)
- **read_external_mesh**: Set to ``.true.`` to use the CUBIT-generated mesh
- **use_existing_STATIONS**: Set to ``.false.`` to generate STATIONS from receiver set parameters
- **External mesh files**: Point to the files in the ``MESH-default`` directory

Receiver Configuration
~~~~~~~~~~~~~~~~~~~~~~

The receivers are defined in the Par_file using the receiver set parameters.
The mesher will automatically generate a STATIONS file based on these parameters:

.. literalinclude:: parameter_files/Par_file
    :caption: Par_file (receiver configuration)
    :language: bash
    :linenos:
    :start-at: # first receiver set
    :end-at: record_at_surface_same_vertical
    :lineno-match:

This configuration creates 11 receivers positioned along a horizontal line at depth
z=3450m, spaced 1000m apart from x=1000m to x=11000m.

Running ``xmeshfem2D``
~~~~~~~~~~~~~~~~~~~~~~

To execute the mesher and generate the database:

.. code:: bash

    xmeshfem2D -p Par_file

This will read the external mesh files and create a binary database file
(``OUTPUT_FILES/database.bin``) that SPECFEM++ can use for the simulation.

Check the mesher generated files in the ``OUTPUT_FILES`` directory:

.. code:: bash

    ls -ltr OUTPUT_FILES

You should see ``database.bin`` and ``STATIONS`` files, along with VTK files
for visualization.

Defining sources
----------------

Next we define the source using a YAML file. For full description on parameters
used to define sources refer :ref:`source_description`.

.. literalinclude:: parameter_files/sources.yaml
    :caption: sources.yaml
    :language: yaml
    :emphasize-lines: 3-6,12-13

In this file, we define a single force source at coordinates (5000.0, 3450.0) meters.
The source uses a Ricker wavelet with a peak frequency of 5.0 Hz, which is appropriate
for this model given its complex structure and heterogeneity.

Configuring the solver
----------------------

Now that we have generated the mesh database and defined the sources, we need to
set up the solver. To do this we define another YAML file ``specfem_config.yaml``.
For full description on parameters used to configure the solver refer
:ref:`parameter_documentation`.

.. literalinclude:: parameter_files/specfem_config.yaml
    :caption: specfem_config.yaml
    :language: yaml
    :emphasize-lines: 18-27,32-34,58

Key configuration points for the Marmousi2 simulation:

- **Time step**: ``dt: 5.0e-6`` (5 microseconds) - small enough for numerical stability
- **Number of steps**: ``nstep: 1000`` - total simulation time of 0.005 seconds
- **Quadrature**: ``GLL4`` - 4th order Gauss-Lobatto-Legendre quadrature
- **Time scheme**: ``Newmark`` - second-order accurate time integration
- **Output format**: ``ascii`` for seismograms

.. note::

    The small time step (5 microseconds) is necessary due to the fine mesh resolution
    and high velocities in the Marmousi2 model. For longer simulations, you may want
    to increase ``nstep`` accordingly.

Running the solver
------------------

Finally, to run the SPECFEM++ solver:

.. code:: bash

    specfem2d -p specfem_config.yaml

.. note::

    Make sure either you are in the executable directory of SPECFEM++ or the
    executable directory is added to your ``PATH``.

The solver will output progress information and save seismograms to
``OUTPUT_FILES/results/``.

Visualizing seismograms
-----------------------

Let us now plot the traces generated by the solver using ``obspy``. The following
Python script reads the ASCII seismogram files and creates plots for each component.

.. literalinclude:: parameter_files/plot_traces.py
    :language: python

To run the plotting script:

.. code:: bash

    python plot_traces.py

This will display the seismograms for both X and Z components.

Expected Results
~~~~~~~~~~~~~~~~

.. figure:: traces_X.svg
   :alt: X-component traces
   :width: 800
   :align: center

   X-component seismograms from the Marmousi2 simulation

.. figure:: traces_Z.svg
   :alt: Z-component traces
   :width: 800
   :align: center

   Z-component seismograms from the Marmousi2 simulation

The seismograms show complex waveforms resulting from scattering and reflection
off the heterogeneous velocity structure in the Marmousi2 model. You should observe:

* Direct P-wave arrivals
* Multiple reflected and converted phases
* Complex coda due to scattering from velocity heterogeneities
* Amplitude variations across receivers due to focusing and defocusing effects

[Optional] Visualizing wavefield snapshots
------------------------------------------

If you enabled the display section output in the ``specfem_config.yaml``, you
should be able to visualize wavefield snapshots using the built in VTK plotter.

The output snapshots will be saved in the ``OUTPUT_FILES/display/`` directory as
PNGs. If you coalesce them into a video using ``ffmpeg``, the resulting wavefield
should look like this:

.. video:: marmousi.mp4
   :width: 800
   :alt: Wavefield propagation through the Marmousi2 model
   :align: center

   Wavefield propagation through the Marmousi2 model. The source is an explosive
   moment tensor located at (5000m, 3450m) with 5 Hz Ricker wavelet as source
   time function. We plot the magnitude of the displacement field.

About the Marmousi2 Model
-------------------------

The Marmousi2 model is an updated version of the original Marmousi model,
designed to be a more realistic representation of geological structures found in
sedimentary basins. It features:

* Complex layered structures with faults and unconformities
* Realistic velocity variations (ranging from ~1500 m/s to ~4500 m/s)
* Variable mesh resolution to capture fine-scale features
* Challenging geometry for testing numerical wave propagation codes
* Ocean layer at the top

The mesh used in this example was improved by Hom Nath Gharti to provide better
element quality and accuracy compared to the original mesh. The CUBIT meshing
workflow, streamlined by Daniel Peter using Python scripts, allows for
reproducible and high-quality mesh generation. See the
:repo-file:`benchmarks/src/dim2/marmousi` for more details on the meshing
process.
