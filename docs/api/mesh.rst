Specifications of the mesh interface
=====================================

In SPECFEM we define the mesh as a C++ struct. The mesh struct defines all the variables nacessary to compute mass and stiffness matrices. The mesh is decomposed into several logical structs which help keep the code concise/readable/maintainable. Below we define the mesh definition and its components.

For performance reasons (specifically on GPUs), the mesh is mostly defined as struct of arrays.

Mesh struct defintion:

.. note::
    ToDo: We might not need nproc in mesh struct

.. code-block:: C++

    struct mesh {
        int npgeo; // Total number of spectral element control nodes
        int nspec; // Total number of spectral elements
        int nproc; // Total number of processors
        specfem::HostView2d coorg; // (x,z) for every spectral element control node

        // Defines type of PML
        // if (region_CPML(ispec) == 0) means ispec is not a PML element
        // if (region_CPML(ispec) == 1) means ispec is a X PML element
        // if (region_CPML(ispec) == 2) means ispec is a Z PML element
        // if (region_CPML(ispec) == 3) means ispec is a XZ PML element
        specfem::HostView1d region_CPML

        // Defines material specfication
        // std::vector<materials> materials[kmato(ispec)] defines the material
        // specification of ispec element
        specfem::HostView1d kmato;

        // Defines global control element number for every control node
        // ipgeo = knods(ia, ispec) where
        //                          0 <= ipgeo < npgeo
        //                          0 <= ia < NGNOD(number of nodes per spectral element)
        //                          0 <= ispec < nspec
        specfem::HostView1d knods;

        // Rest of the definitions are given below
        interface inter;
        absorbing_boundary abs_boundary;
        prop properties;
        acoustic_free_surface acfree_surface;
        forcing_boundary acforcing_boundary;
        tangential_elements tangential_nodes;
        axial_elements axial_nodes;
    };


The `interface struct` defines the variables needed to compute MPI buffers.

.. note::
    ToDo: Document MPI interfaces

The `properties struct` defines mesh properties

.. code-block:: C++

    struct prop {
        int numat; // Total number of material types
        int ngnod; // number of nodes per spectral element
        int nspec; // Total number of spectral elements
        int pointsdisp; // Total number of points to display (Only used for visualization)
        int nelemabs; // Total number of absorbing elements
        int nelem_acforcing; // Total number of acoustic forcing elements
        int nelem_acoustic_surface; // Total number of acoustic surfaces
        int num_fluid_solid_edges; // Total number of elements on fluid solid boundary
        int num_fluid_poro_edges; // Total number of elements on fluid porous boundary
        int num_solid_poro_edges; // Total number of elements on solid solid boundary
        int nnodes_tagential_curve; // Total number of elements on tangential curve
        int nelem_on_the_axis; // Total number of axial elements
        bool plot_lowerleft_corner_only;
    };

The `absorbing_boundary struct` defines the variables needed to impose stacey absorbing boundary conditions

.. code-block:: C++

    struct absorbing_boundary {

        // numsabs(i) defines the ispec of ith absorbing element
        specfem::HostView1d numabs;

        // Defines if the absorbing boundary type is top, left, right or bottom
        // This is only used during plotting
        specfem::HostView1d abs_boundary_type;

        // Here
        //      edge1 as the bottom boundary
        //      edge2 as the right boundary
        //      edge3 as the top boundary
        //      edge4 as the left boundary

        // ibegin_<edge#> defines the i or j index limits for loop iterations
        // Check demostration code below
        specfem::HostView1d ibegin_edge1, ibegin_edge2, ibegin_edge3, ibegin_edge4;
        specfem::HostView1d iend_edge1, iend_edge2, iend_edge3, iend_edge4;

        // Specifies if an element is bottom, right, top or left absorbing boundary
        // for bottom boundary
        //          codeabs(i, 0) == true
        // for right boundary
        //          codeabs(i, 1) == true
        // for top boundary
        //          codeabs(i, 2) == true
        // for left boundary
        //          codeabs(i, 3) == true
        specfem::HostView2d<bool> codeabs;

        // Specifies if an element is bottom-left, bottom-right, top-left or top-right
        // corner element
        // for bottom-left boundary element
        //          codeabscorner(i, 0) == true
        // for bottom-right boundary element
        //          codeabscorner(i, 1) == true
        // for top-left boundary element
        //          codeabscorner(i, 2) == true
        // for top-right boundary element
        //          codeabscorner(i, 3) == true
        specfem::HostView2d<bool> codeabscorner;

        // Specifies the ispec_edge for that edge
        // For example
        //      ib_bottom(i) = ispec_bottom
        //          where 0 < ispec_bottom < total number of absorbing boundary elements on
        //                                   the bottom boundary

        // This can specifically used to store data related to absorbing elements in a
        // compact format
        specfem::HostView1d ib_bottom, ib_top, ib_right, ib_left;
    };

The following code snippet demostrates the usage of absorbing boundary struct to impose stacey boundary conditions

.. note::
    Todo - Add code snippet to demostrate absorbing_boundary struct

The `forcing_boundary struct` specifies the variables required to impose acoustic forcing at rigid boundaries

.. code-block:: C++

    struct forcing_boundary {

        // numacforcing(i) defines the ispec of ith acoustic forcing element
        specfem::HostView1d numacforcing;

        // Defines if the acoustic forcing boundary type is top, left, right or bottom
        // This is only used during plotting
        specfem::HostView1d typeacforcing;

        // Here
        //      edge1 as the bottom boundary
        //      edge2 as the right boundary
        //      edge3 as the top boundary
        //      edge4 as the left boundary

        // ibegin_<edge#> defines the i or j index limits for loop iterations
        // Check demostration code below
        specfem::HostView1d ibegin_edge1, ibegin_edge2, ibegin_edge3, ibegin_edge4;
        specfem::HostView1d iend_edge1, iend_edge2, iend_edge3, iend_edge4;

        // Specifies if an element is bottom, right, top or left acoustic forcing boundary
        // for bottom boundary
        //          codeacforcing(i, 0) == true
        // for right boundary
        //          codeacforcing(i, 1) == true
        // for top boundary
        //          codeacforcing(i, 2) == true
        // for left boundary
        //          codeacforcing(i, 3) == true
        specfem::HostView2d<bool> codeacforcing;

        // Specifies the ispec_edge for that edge
        // For example
        //      ib_bottom(i) = ispec_bottom
        //          where 0 < ispec_bottom < total number of acoustic forcing elements on
        //                                   the bottom boundary

        // This can specifically used to store data related to acoustic forcing elements in a
        // compact format
        specfem::HostView1d ib_bottom, ib_top, ib_right, ib_left;
    };

The following code snippet demostrates the usage of acoustic forcing boundary struct to impose stacey boundary conditions

.. note::
    Todo - Add code snippet to demostrate acforcing_boundary struct

.. note::
    Todo - Add acfree_surface documentation

.. note::
    Todo - Add tangential surface elements documentation

The `axial_elements struct` defines if an element is axial or not

.. code-block:: C++

    struct axial_elements {
        // is_on_the_axis(ispec) defines is an element is axial or not
        specfem::HostView1d<bool> is_on_the_axis;
    };
