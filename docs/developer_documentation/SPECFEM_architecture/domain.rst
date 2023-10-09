.. domain_dev_gui::

Domain developer guide
======================

`specfem::domain::domain` is a templated C++ class. A templated domain class allows us to provide cookie-cutter parallelism frameworks while allowing developers to describe the physics at elemental level `specfem::domain::impl::elements`. This developer guide provides an in-depth methodology for understanding and extending the domain class to implement new physics.

Brief introduction to C++ templates
-----------------------------------

C++ templates are a powerful tool for generic programming. They allow us to write code that is independent of the type of data it operates on.

.. code-block:: C++

    template <typename T>
    T add(T a, T b) {
        return a + b;
    }

For example the function above, adds two numbers without knowing whether they are integers, floating point numbers, or complex numbers. The compiler will generate a different version of the function for each type of data. Thus the compiler needs to know the type of data at compile time. In modern compilers (C++17 and above) the compiler can utilize type deduction to infer the type of data from the function arguments.

The power of templates for writing portable code becomes obvious when using user defined types. For example, consider the following function which allows us to calculate L2 norm of a vector in dimension independent manner.

.. code-block:: C++

    class dim2{
    public:
        constexpr int dim = 2;
    };

    class dim3{
    public:
        constexpr int dim = 3;
    };

    template <typename T>
    double l2_norm(const std::vector<double> vec) {
        double norm = 0.0;
        assert(vec.size() == T::dim;
        for (int i = 0; i < T::dim; i++) {
            norm += vec[i] * vec[i];
        }
        return std::sqrt(norm);
    }

There are a couple of key points to note here from a performance standpoint:

1. The compiler will generate a different version of the function for each dimension.
2. Since `T::dim` is a `constexpr` the compiler will unroll the loop - which on modern CPUs and GPUs can lead to significant performance gains.

Apart from performance, templates also provide us a way to define a generic interface for different types of data i.e. in the above code we didn't need to write two different functions for `dim2` and `dim3` vectors. The importance of this in SPECFEM context will become clear in the following sections.

Anatomy of a SPECFEM domain
---------------------------

The following figure shows the different components of a SPECFEM domain.

.. warning::

    Need to add a figure here


As the name suggests `specfem::domain::domain` is closely related to a spectral element domain. The domain is comprised of set of finite elements. The finite element method provide us a way to descritize the domain into small elements where we can approximate the solution using a polynomial basis. The approach is then to compute the coefficients of the polynomial basis at elemental levels which greatly reduces the computational cost.

Let us look at computing the contribution of acoustic domain to global :math:`\frac{{\partial \chi}{\partial t^2}}`. The methematical formulation to which is given by (Komatitsch and Tromp, 2002 :cite:`Komatitsch2002`):

.. warning::

    Need to add the equation here

The key thing to note in the above expression is, given the types of elements inside a domain we can loop over all the elements and compute its contribution in an element agnostic way. This is the key idea behind the domain class. The domain class provides a generic interface to compute the elemental contribution of a given physics. This lets us separate the physics from the parallelism.

Understanding the parallelism
------------------------------

Let us now look at a naive serial implementation for the above formulation in 3D.

.. code:: C++

    void compute_acoustic_stiffness_interaction() {
        for (int ispec = 0; ispec < nspec; i++) {
            for (int iz = 0; iz < ngllz; iz++) {
                for (int iy = 0; iy < ngllz; iy++) {
                    for (int ix = 0; ix < ngllx; ix++) {
                        // compute the global index of the GLL point
                        iglob = ibool(ispec, iz, iy, ix); // ibool is the mapping vector from GLL point to global index
                        // compute gradient at GLL point ix, iy, iz
                        acoustic_element.compute_gradient(ix, iy, iz);
                        // compute stresses at GLL point ix, iy, iz
                        acoustic_element.compute_stresses(ix, iy, iz);
                        // compute the md2chidt2 at GLL point ix, iy, iz
                        type_real md2chidt2 = acoustic_element.compute_acceleration(ix, iy, iz);
                        // add the contribution to the global vector
                        potential_dot_dot[iglob] += md2chidt2;
                    }
                }
            }
        }
    }

Since the computations in each dimension are independent of each other we can simplify the above code even further.

.. code:: C++

    void compute_acoustic_stiffness_interaction() {
        for (int ispec = 0; ispec < nspec; i++) {
            for (int xyz = 0; xyx < ngllxyz; xyx++) {
               auto [ix, iy, iz] = sub2ind(xyz);
               // compute the global index of the GLL point
               iglob = ibool(ispec, iz, iy, ix); // ibool is the mapping vector from GLL point to global index
               // compute gradient at GLL point ix, iy, iz
               acoustic_element.compute_gradient(ix, iy, iz, <other_arguments>);
               // compute stresses at GLL point ix, iy, iz
               acoustic_element.compute_stresses(ix, iy, iz, <other_arguments>);
               // compute the md2chidt2 at GLL point ix, iy, iz
               type_real md2chidt2 = acoustic_element.compute_acceleration(ix, iy, iz, <other_arguments>);
               // add the contribution to the global vector
               potential_dot_dot[iglob] += md2chidt2;
            }
        }
    }

Now let us template the above code to make it dimension independent using a bit of macro magic.

.. code:: C++

    #ifdef DIM2
        #define INDEX iz,ix
    #endif

    #ifdef DIM3
        #define INDEX iz,iy, ix
    #endif

    template <typename quadrature_points>
    void compute_acoustic_stiffness_interaction() {
        for (int ispec = 0; ispec < nspec; i++) {
            for (int qp = 0; qp < dimension::get_num_qp(); qp++) {
                auto [INDEX] = sub2ind(qp);
                // rest of the code
                ...
            }
        }
    }

Kokkos parallelism
...................

The above code is a good starting point for parallelizing the code. A naive method of parallelizing the above section would be to distribute the 2 for loops among the available threads for example using OpenMP `collapse(2)` clause. However, since different elements could have different implementation (physics) for calculating the gradient, stresses, and acceleration contribution such a parallelization would result in poor performance on GPUs cause of warp divergence. Even on CPUs the performance would be poor since compiler could miss vectorization opportunities.

Kokkos provides a natural formalism to exploit this type of parallelism using :ref:`heirarchical parallelism <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism.html>`_ . The idea is to parallelize the outer loop over elements using Kokkos teams and then parallelize the inner loop over quadrature points using Kokkos thread teams. This guarantees that all the threads in a team (which is mapped to CUDA blocks on NVIDIA GPUs) execute the same code path - thus avoiding warp divergence.

.. code:: C++

    template <typename quadrature_points>
    void compute_acoustic_stiffness_interaction() {
        Kokkos::parallel_for("compute_acoustic_stiffness_interaction", Kokkos::TeamPolicy<execution_space>(nspec, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<execution_space>::member_type& team) {
            int ispec = team.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dimension::get_num_qp()), [=] (const int& qp) {
                auto [INDEX] = sub2ind(qp);
                // rest of the code
                ...
            });
        });
    }

Optimizing using shared/cache memory
....................................

At this point, it would be good to look at elmental implementations to understand the performance bottlenecks. Let us start by looking at function to compute the gradient of the potential inside a 2D acoustic element.

.. code:: C++

    class acoustic_element {
        void compute_gradient(
            const int &ispec, const int &xz, const View2d<type_real> hprime_xx,
            const View2d<type_real> hprime_zz, const View1d<type_real> field_chi,
            type_real *dchidxl, type_real *dchidzl){


            int ix, iz, iglob;
            sub2ind(xz, NGLL, iz, ix);

            const type_real xixl = this->xix(ispec, iz, ix);
            const type_real gammaxl = this->gammax(ispec, iz, ix);
            const type_real xizl = this->xiz(ispec, iz, ix);
            const type_real gammazl = this->gammaz(ispec, iz, ix);

            type_real dchi_dxi = 0.0;
            type_real dchi_dgamma = 0.0;

            for (int l = 0; l < ngllx; l++) {
                iglob = ibool(ispec, iz, l)
                dchi_dxi += hprime_xx(ix, l) * field_chi(iglob, 0);
            }

            for (int l = 0; l < ngllz; l++) {
                iglob = ibool(ispec, l, ix)
                dchi_dgamma += hprime_zz(iz, l) * field_chi(iglob, 0);
            }

            // dchidx
            dchidxl[0] = dchi_dxi * xixl + dchi_dgamma * gammaxl;

            // dchidz
            dchidzl[0] = dchi_dxi * xizl + dchi_dgamma * gammazl;

            return;
        }
    };

This implementation is not very efficient since it requires a lot of global memory accesses. In particular, if we look at the inner loop the accesses to `hprime_xx`, `hprime_zz` and `field_chi` are not coalesced. To improve the performance we can use shared memory to cache the values of `hprime_xx`, `hprime_zz` and `field_chi` for each element.

.. code:: C++

    template <typename quadrature_points>
    void compute_acoustic_stiffness_interaction() {

        // allocate shared memory
        typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
        // Define a view type in ScratchSpace
        typedef Kokkos::View<type_real**,ScratchSpace,
                    Kokkos::MemoryTraits<Kokkos::Unmanaged>> scratch_view;

        // allocate shared memory for hprime_xx, hprime_zz, and field_chi
        size_t scratch_size =
                    scratch_view::shmem_size(ngllx, ngllx) +
                    scratch_view::shmem_size(ngllx, ngllx) +
                    scratch_view::shmem_size(ngllz, ngllx);

        int scratch_size =
        Kokkos::parallel_for("compute_acoustic_stiffness_interaction", Kokkos::TeamPolicy<execution_space>(nspec, Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<execution_space>::member_type& team) {
            int ispec = team.league_rank();
            // allocate shared memory
            scratch_view s_hprime_xx(team.team_scratch(0), ngllx, ngllx);
            scratch_view s_hprime_zz(team.team_scratch(0), ngllz, ngllz);
            scratch_view s_field_chi(team.team_scratch(0), ngllz, ngllx);

            // copy data to shared memory
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ngllxx), [=] (const int &xx) {
                i = xx % ngllx;
                j = xx / ngllx;
                s_hprime_xx(i, j) = hprime_xx(i, j);
            });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ngllzz), [=] (const int &xx) {
                i = zz % ngllz;
                j = zz / ngllz;
                s_hprime_zz(i, j) = hprime_zz(i, j);
            });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ngllxz) [=] (const int &xz) {
                int ix, iz;
                sub2ind(xz, ngllxz, iz, ix);
                s_field_chi(iz, ix) = field_chi(ibool(ispec, iz, ix), 0);
            });

            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dimension::get_num_qp()), [=] (const int& qp) {
                auto ix, iz = sub2ind(qp);

                acoustic_element.compute_gradient(ispec, qp, s_hprime_xx, s_hprime_zz, s_field_chi, dchidxl, dchidzl);
                // rest of the code
                ...
            });
        });
    }

.. note::

    The description provided here serves as a good starting point for understanding the domain class. The actual implementation, while based on ideas presented here, is more complex and optimized for performance.

Specializing elemental implementations
--------------------------------------

Let us next conside the elemental implementation for computing stresses in 2D isotropic elastic element and 2D anisotropic elastic element. The stress tensor for a general element is given by:

.. warning::

    Need to add the equation here

However, for isotropic elements the tensor :math:`\bf{c(x)}` is a diagonal with only 2 independent components. Thus the stress tensor simplifies to:

.. warning::

    Need to add the equation here

Computationally, the number of accesses from global memory when computing the stresses for isotropic elements is an order or magnitude less than that for anisotropic elements (2 accesses vs 27 accesses). Thus it makes sense to specialize the elemental implementation for isotropic and anisotropic elements.

.. code:: C++

    // definition of element class
    template <typename... properties>
    class element{}

    // specialization for acoustic isotropic elements
    template <>
    class element<dim2, acoustic, isotropic>{
        // implementation specific details
    }

    // specialization for elastic isotropic elements
    template <>
    class element<dim2, elastic, isotropic>{
        // implementation specific details
    }

    // specialization for elastic anisotropic elements
    template <>
    class element<dim2, elastic, anisotropic>{
        // implementation specific details
    }

Using the above specialization we've provided a unified interface for acoustic and elastic elements where we can further specialize those elements based on domain/spectral element properties.

.. note::

    Specializing the elemental implementation for different types of elements is a powerful tool for performance optimization. However, it requires us to launch a different kernel for each type of element. This creates a bookkeeping overhead - where we need to make sure every element is accounted for exactly once. The launch of the kernels itself is done using `specfem::domain::impl::kernels`

.. note::

    The other solution to the above problem is to use a single kernel and use class inheritance and polymorphism to deduce elemental specialization at runtime. However, this approach is not very efficient since GPUs are very inefficient at resolving virtual function calls.

Optimization using loop unrolling
---------------------------------
