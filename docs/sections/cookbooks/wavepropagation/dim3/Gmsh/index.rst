.. _docs_sections_cookbooks_dim3_Gmsh:

Using a Gmsh mesh in SPECFEM++ (3D)
=====================================

As with any finite element solver, SPECFEM++ requires a mesh generation step to
discretize the domain. The spectral element solver requires a hexahedral mesh and
does not support tetrahedral meshes. In this example we use
`Gmsh <https://gmsh.info>`_ — a free, cross-platform mesh generator — to create a
simple 3D box domain, export it using a Python script, and run the simulation using
the ``xdecompose_mesh`` partitioner followed by ``specfem 3d``.

.. seealso::

   :ref:`known_limitations` — several 3-D features (PML boundaries,
   poroelastic/anisotropic materials, GLL-level property input/output) are not yet
   implemented and will raise a runtime error if requested.

.. note::

    This cookbook is not an in-depth Gmsh tutorial. Its goal is to show how to
    define a Gmsh mesh so that SPECFEM++ can read it. For more information on
    meshing with Gmsh please refer to the
    `Gmsh tutorials <https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-tutorial>`_.

.. note::

    All cookbook files can be copy and pasted from the code blocks below, or you
    can download a zip file containing all the files needed to run this example.

    .. download-folder:: parameter_files
        :filename: gmsh_3d_cookbook.zip
        :text: GMSH 3D cookbook

Setting up your workspace
-------------------------

Install Gmsh and its Python API (if not already available):

.. code-block:: bash

    pip install gmsh

Create a working directory and make sure the SPECFEM++ executables are on your
``PATH``.

.. code-block:: bash

    mkdir -p ~/specfempp-examples/gmsh-halfspace-3d
    cd ~/specfempp-examples/gmsh-halfspace-3d

.. code-block:: bash

    which specfem

If the command does not return a path, add the executable directory to ``PATH``:

.. code-block:: bash

    export PATH=$PATH:<PATH TO SPECFEM++ DIRECTORY>/bin

Create the output directories and placeholder files:

.. code-block:: bash

    mkdir -p MESH
    mkdir -p DATABASES_MPI
    mkdir -p OUTPUT_FILES/results
    mkdir -p DATA
    touch specfem_config.yaml
    touch sources.yaml

Creating the geometry
---------------------

We will create a 10 km × 10 km × 5 km box domain with the free surface at z = 0 and
the bottom of the model at z = -5000 m.

Save the following as ``create_mesh.py`` in your working directory:

.. code-block:: python
    :caption: create_mesh.py

    import gmsh

    gmsh.initialize()
    gmsh.model.add("halfspace_3d")

    ## Create a 10 km x 10 km x 5 km box: x:[0,10000], y:[0,10000], z:[-5000,0]
    box = gmsh.model.occ.addBox(0, 0, -5000, 10000, 10000, 5000)
    gmsh.model.occ.synchronize()

    ## ---------------------------------------------------------------
    ## Physical groups
    ## Volume group for the elastic material
    ## ---------------------------------------------------------------
    gmsh.model.addPhysicalGroup(3, [box], name="material_1")

    ## ---------------------------------------------------------------
    ## Surface groups for boundary conditions
    ## Use the bounding box of each surface to identify the face.
    ## ---------------------------------------------------------------
    eps = 1.0
    for dim, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        if xmax < eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_xmin")
        elif xmin > 10000 - eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_xmax")
        elif ymax < eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_ymin")
        elif ymin > 10000 - eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_ymax")
        elif zmax < -5000 + eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_bottom")
        else:                ## z = 0: free surface
            gmsh.model.addPhysicalGroup(2, [tag], name="free_or_abs_zmax")

    ## ---------------------------------------------------------------
    ## Structured transfinite hex mesh at 500 m element size
    ## ---------------------------------------------------------------
    ## Number of nodes along each edge (nodes = elements + 1)
    n_horiz = 21   ## 20 elements across 10 km  = 500 m
    n_vert  = 11   ## 10 elements across  5 km  = 500 m

    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        length = max(abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin))
        n = round(length / 500) + 1
        gmsh.model.mesh.setTransfiniteCurve(tag, n)

    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(dim, tag)

    gmsh.model.mesh.setTransfiniteVolume(box)

    gmsh.model.mesh.generate(3)
    gmsh.write("halfspace.msh")
    gmsh.finalize()
    print("Mesh written to halfspace.msh")

Run the script to generate the mesh file:

.. code-block:: bash

    python create_mesh.py

After the script completes you should see ``halfspace.msh`` in the working
directory. You can open it in the Gmsh GUI to inspect the mesh:

.. code-block:: bash

    gmsh halfspace.msh

.. note::

    The transfinite structured meshing algorithm requires that opposite faces of
    the volume are meshed with the same topology. For a simple box this is
    guaranteed automatically. For more complex geometries you may need to specify
    the source and target surfaces for the transfinite sweep manually using
    ``gmsh.model.mesh.setTransfiniteVolume(tag, corner_points)``.

After meshing you should see approximately **4 000 hexahedral elements**
(20 × 20 × 10).

Understanding physical groups
------------------------------

Physical groups in Gmsh serve the same role as block names in CUBIT: they tag
mesh entities so the export script knows how to route them to the correct output
files. The naming convention mirrors the CUBIT workflow exactly:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Group name (dim)
     - Output file
   * - ``material_1`` (dim=3)
     - element connectivity, material mapping, velocity file
   * - ``abs_xmin`` (dim=2)
     - ``absorbing_surface_file_xmin``
   * - ``abs_xmax`` (dim=2)
     - ``absorbing_surface_file_xmax``
   * - ``abs_ymin`` (dim=2)
     - ``absorbing_surface_file_ymin``
   * - ``abs_ymax`` (dim=2)
     - ``absorbing_surface_file_ymax``
   * - ``abs_bottom`` (dim=2)
     - ``absorbing_surface_file_bottom``
   * - ``free_or_abs_zmax`` (dim=2)
     - ``free_or_absorbing_surface_file_zmax``

.. tip::

    The bounding-box approach in ``create_mesh.py`` automatically identifies
    each face regardless of the internal surface tag numbers Gmsh assigns.
    This makes the script robust to geometry modifications.

Exporting the mesh
------------------

The export script is available in the SPECFEM++ repository at
``<PATH TO SPECFEM++ DIRECTORY>/scripts/export_gmsh3d.py``.

Unlike the CUBIT export script, ``export_gmsh3d.py`` is a standard Python script
that reads the saved ``.msh`` file — it does not need to run inside Gmsh.

Add it to your path or copy it to your working directory, then run:

.. code-block:: python
    :caption: export.py

    import sys
    sys.path.insert(0, "<PATH TO SPECFEM++ DIRECTORY>/scripts")

    from export_gmsh3d import export2SPECFEM3D_gmsh, MaterialSpec

    ## Material properties for material_1 (elastic granite)
    materials = {
        1: MaterialSpec(
            mat_id    = 1,
            domain_id = 2,       ## 2 = elastic
            rho       = 2700.0,  ## density (kg/m^3)
            vp        = 6000.0,  ## P-wave velocity (m/s)
            vs        = 3500.0,  ## S-wave velocity (m/s)
            q_kappa   = 9999.0,  ## no attenuation
            q_mu      = 9999.0,
        ),
    }

    export2SPECFEM3D_gmsh("halfspace.msh", materials, outdir="MESH")

.. code-block:: bash

    python export.py

Alternatively, use the command-line interface directly:

.. code-block:: bash

    python <PATH TO SPECFEM++ DIRECTORY>/scripts/export_gmsh3d.py \
        halfspace.msh MESH \
        --mat 1 2 2700 6000 3500

This will print status messages and generate the following files inside the ``MESH``
directory:

- ``nodes_coords_file`` — node coordinates
- ``mesh_file`` — hexahedral element connectivity
- ``materials_file`` — element-to-material mapping
- ``nummaterial_velocity_file`` — material velocity properties
- ``absorbing_surface_file_xmin`` — xmin absorbing face elements
- ``absorbing_surface_file_xmax`` — xmax absorbing face elements
- ``absorbing_surface_file_ymin`` — ymin absorbing face elements
- ``absorbing_surface_file_ymax`` — ymax absorbing face elements
- ``absorbing_surface_file_bottom`` — bottom absorbing face elements
- ``free_or_absorbing_surface_file_zmax`` — top free surface elements

.. note::

    The ``domain_id`` in ``MaterialSpec`` controls the physics:
    ``1`` = acoustic (fluid), ``2`` = elastic (solid).
    Set ``vs = 0`` for acoustic materials.

Running ``xdecompose_mesh``
---------------------------

``xdecompose_mesh`` partitions the mesh text files into a binary database that
``specfem 3d`` can read. We first define the decomposer parameters in a ``Par_file``.

.. literalinclude:: parameter_files/Par_file
    :language: bash
    :caption: Par_file

Key parameters to note:

- ``NPROC`` — number of MPI partitions. Set to 1 here for a serial run. Increase it
  to match the number of MPI processes you plan to use with ``specfem 3d``.
- ``NGNOD = 8`` — matches our HEX8 mesh.
- ``LOCAL_PATH`` — directory where the binary partition databases are written.
- The ``NODES_COORDS_FILE`` through ``FREE_OR_ABSORBING_SURFACE_FILE_ZMAX`` entries
  point to the files generated by ``export2SPECFEM3D_gmsh`` in the previous step.

Run the decomposer from your working directory:

.. code-block:: bash

    xdecompose_mesh -p Par_file

After a successful run you should see a binary database file inside ``DATABASES_MPI``:

.. code-block:: bash

    ls DATABASES_MPI/

For ``NPROC = 1`` this produces ``DATABASES_MPI/Database.bin``. For ``NPROC > 1``
a subdirectory ``DATABASES_MPI/Database/`` is created containing one
``proc_<rank>.bin`` file per MPI rank.

Defining receivers (stations)
-------------------------------

Receivers are defined in the ``DATA/STATIONS`` file. Each line specifies one station
with the format:

``STATION_NAME NETWORK_CODE X Y ELEVATION BURIAL``

.. literalinclude:: parameter_files/DATA/STATIONS
    :language: bash
    :caption: STATIONS

These six stations are distributed along the free surface (z = 0), displaced 1–3 km
from the source in the x direction.

Defining sources
----------------

The source is defined in ``sources.yaml``. We place a vertical force at the centre of
the domain, 500 m below the free surface, with a 1 Hz Ricker wavelet.

.. literalinclude:: parameter_files/sources.yaml
    :language: yaml
    :caption: sources.yaml

Configuring the solver
-----------------------

Define the simulation parameters in ``specfem_config.yaml``:

.. literalinclude:: parameter_files/specfem_config.yaml
    :language: yaml
    :caption: specfem_config.yaml
    :emphasize-lines: 20-24,41-42

A few parameters to highlight:

- ``dt = 0.004`` s and ``nstep = 1250`` gives 5 s of simulation, enough to record
  both the direct P and S arrivals at all stations.  The CFL stability limit for
  this mesh (500 m elements, GLL4, vp = 6000 m/s) is approximately
  0.3 × 86 m / 6000 m/s ≈ 0.0043 s, so ``dt = 0.004`` s sits comfortably
  within the stable range.
- ``mesh-database`` points to ``DATABASES_MPI/Database.bin``, which is where
  ``xdecompose_mesh`` writes the single-process partition database. If you used
  ``NPROC > 1`` you do not need to change this path — ``specfem`` automatically
  appends the rank suffix.

Running the solver
------------------

.. code-block:: bash

    specfem 3d -p specfem_config.yaml

The solver writes ASCII seismogram files for each station and component to
``OUTPUT_FILES/results/``.

.. code-block:: bash

    ls OUTPUT_FILES/results/

You should see one file per station per displacement component, e.g.:
``DB.ST001.BXX.semd``, ``DB.ST001.BXY.semd``, ``DB.ST001.BXZ.semd``.

[Optional] Visualizing the wavefield
--------------------------------------

To visualize the wavefield propagation SPECFEM++ must be built with VTK and HDF5
support. Uncomment the ``display`` section in ``specfem_config.yaml``:

.. code-block:: yaml

    display:
      format: VTKHDF
      directory: OUTPUT_FILES/
      field: displacement
      component: z
      simulation-field: forward
      time-interval: 5

Rerun the solver, then open Paraview and load the generated VTK files from the
``OUTPUT_FILES`` directory to visualize the wavefield propagation.
