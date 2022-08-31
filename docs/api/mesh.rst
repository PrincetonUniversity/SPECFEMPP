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
        // std::vector<materials> materials[kmato(ispec)] defines the material specification of ispec element
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
