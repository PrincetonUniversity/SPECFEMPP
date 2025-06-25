.. _data_container:

DATA_CONTAINER
==============

Overview
--------

The ``DATA_CONTAINER`` macro is a powerful utility for generating standard data container structures for different medium types in SPECFEMPP. It creates a structured collection of ``DomainView2d`` elements with consistent interface methods for accessing and synchronizing data between host and device memory spaces.

Syntax
------

.. code-block:: cpp

   DATA_CONTAINER(param1, param2, ...)

Where ``param1``, ``param2``, etc. are the names of the ``DomainView2d`` elements to be included in the container.

Description
-----------

The ``DATA_CONTAINER`` macro generates:

1. A set of ``DomainView2d`` fields with the specified names
2. Corresponding host mirror variables for each field
3. Constructors that initialize these views with the proper dimensions
4. Methods to iterate over elements on both device and host
5. Synchronization methods to copy data between device and host views
6. Methods to iterate over all device and host views

Generated Methods
-----------------

+-------------------------------+-------------------------------------------------------+
| Method                        | Description                                           |
+===============================+=======================================================+
| ``copy_to_device()``          | Synchronizes all data from host to device             |
+-------------------------------+-------------------------------------------------------+
| ``copy_to_host()``            | Synchronizes all data from device to host             |
+-------------------------------+-------------------------------------------------------+
| ``for_each_on_device(index,`` | Applies a functor to each element specified by an     |
| ``functor)``                  | index on the device                                   |
+-------------------------------+-------------------------------------------------------+
| ``for_each_on_host(index,``   | Applies a functor to each element specified by an     |
| ``functor)``                  | index on the host                                     |
+-------------------------------+-------------------------------------------------------+
| ``for_each_device_view(``     | Applies a functor to each device view                 |
| ``functor)``                  |                                                       |
+-------------------------------+-------------------------------------------------------+
| ``for_each_host_view(``       | Applies a functor to each host view                   |
| ``functor)``                  |                                                       |
+-------------------------------+-------------------------------------------------------+

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: cpp

   struct MyContainer {
     DATA_CONTAINER(rho, kappa, mu)
   };

This generates a container with three ``DomainView2d`` elements: ``rho``, ``kappa``, and ``mu``.

In Context of SPECFEMPP
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   namespace specfem::medium::properties {
     template <>
     struct data_container<specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic> {
       constexpr static auto dimension = specfem::dimension::type::dim2;
       constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
       constexpr static auto property_tag = specfem::element::property_tag::isotropic;

       DATA_CONTAINER(kappa, mu, rho)
     };
   }

Generated Code Example
~~~~~~~~~~~~~~~~~~~~~~

For ``DATA_CONTAINER(rho, kappa)``, the generated code includes:

.. code-block:: cpp

   // Field definitions
   DomainView2d<type_real, 3, Kokkos::DefaultExecutionSpace::memory_space> rho;
   DomainView2d<type_real, 3, Kokkos::DefaultExecutionSpace::memory_space> kappa;
   typename decltype(rho)::HostMirror h_rho;
   typename decltype(kappa)::HostMirror h_kappa;

   // Constructor
   data_container(const int nspec, const int ngllz, const int ngllx)
       : rho("rho", nspec, ngllz, ngllx),
         kappa("kappa", nspec, ngllz, ngllx),
         h_rho(specfem::kokkos::create_mirror_view(rho)),
         h_kappa(specfem::kokkos::create_mirror_view(kappa)) {}

   // Synchronization methods
   void copy_to_device() {
     specfem::kokkos::deep_copy(rho, h_rho);
     specfem::kokkos::deep_copy(kappa, h_kappa);
   }

   void copy_to_host() {
     specfem::kokkos::deep_copy(h_rho, rho);
     specfem::kokkos::deep_copy(h_kappa, kappa);
   }

   // And other accessor methods...

Use Cases
---------

The ``DATA_CONTAINER`` macro is extensively used throughout SPECFEMPP for various medium types:

- Acoustic media (``rho_inverse``, ``kappa``)
- Elastic isotropic media (``kappa``, ``mu, rho``)
- Elastic anisotropic media (``c11``, ``c13``, ``c15``, ``c33``, ``c35``, ``c55``, ``c12``, ``c23``, ``c25``, ``rho``)
- Elastic isotropic Cosserat media (``rho``, ``kappa``, ``mu``, ``nu``, ``j``, ``lambda_c``, ``mu_c``, ``nu_c``)
- Poroelastic media (``phi``, ``rho_s``, ``rho_f``, ``tortuosity``, ``mu_G``, ``H_Biot``, ``C_Biot``, ``M_Biot``, etc.)

Technical Implementation
------------------------

The macro leverages Boost Preprocessor library to generate repetitive code patterns, enabling compile-time code generation without the need for template metaprogramming. It handles the complex task of creating consistent interfaces for accessing data in a heterogeneous computing environment.
