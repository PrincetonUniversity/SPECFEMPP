.. compute_dev_guide:

Assembly namespace developer guide
==================================

The ``assembly`` namespace in SPECFEM++ is designed to separate simulation data from the C++ methods and objects used to assembly the evolution of the wavefield. This separation allows for a more modular and flexible approach to implementing post-processing and ad-hoc analysis capabilities without modifying the core routines used to assembly the wavefield.

By separating the simulation data from the computation methods, it becomes easier to view and manipulate the data without affecting the underlying computation. This can be especially useful for debugging and testing purposes, as well as for implementing new features and capabilities.

The ``assembly`` struct is key part of the SPECFEM++ codebase, and understanding its interaction with the rest of the code is essential for developing new features and capabilities. This developer guide is intended to provide a high-level overview of the ``assembly`` struct and its interaction with the core SPECFEM++ computational routines.

Understanding Kokkos Views
--------------------------

Before we explain how the data stored in assembly struct is accessed by various SPECFEM++ computational routines. We need to understand a few key concepts of Kokkos views. Kokkos views are multi-dimensional arrays which are used to store/sync data across a host and device. The views are templated C++ classes where the template parameters define, at runtime, the dimension, data type, memory space, and layout of the view. The size of the array can be defined either at compile time or at runtime. For e.g. :

.. code-block:: C++

    // alias for 1D device view of doubles
    using DeviceView1d = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

    // alias for 2D host view of doubles
    using HostView2d = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;

    // 1D view of size 10
    DeviceView1d view_1d("view_1d", 10);

    // 2D view of size 10x10
    HostView2d h_view_2d("view_2d", 10, 10);

Anatomically the view contains 2 elements:
1. The metadata describing the view (e.g. size, layout, memory space, etc.).
2. An allocation (pointer) to the data stored in the view.

.. note::
    The default copy constructor and assignment operator for Kokkos views perform a shallow copy of the view. This means that the metadata is copied, but the data is not. Instead, the data is shared between the original view and the copy. This is important to keep in mind when passing views to functions, as the function may modify the data in the view. This is fundamentally how we pass data stored in ``assembly`` namespace between methods/objects in SPECFEM++.


Adding new data to ``assembly`` namespace
-----------------------------------------

Idea behind ``assembly`` namespace is to provide a data layer to access simulation data during or at the end of simulation. Thus it makes sense to extend the namespace with new data when implemeting new features. A few things to keep in mind while adding new data to ``assembly`` namespace:

1. Create a logical heirarchical structure for the data. For e.g. ``specfem::assembly::receivers`` struct which contains all the data related to receivers. This grouping allows us to pass only the receiver data to a methods/objects which needs it.
2. Think about when to initialize the data and where it will be accessed. Create device views are host mirrors when required and ``deep_copy`` the data to the device views when initialized.
3. GPU memory is precious, do not allocate more memory then required on the device. Do any pre-processing on the host and only copy the data required for computation to the device to reduce memory footprint.
4. Make sure any data that could be accessed in the future is available from within the ``assembly`` namespace.
