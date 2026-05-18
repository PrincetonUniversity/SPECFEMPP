#include "compute_factors.tpp"
#include "compute_tau_eps.tpp"
#include "compute_tau_sigma.tpp"
#include "specfem/attenuation/compute_factors.hpp"
#include "specfem/attenuation/compute_tau_eps.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"

template specfem::attenuation::AttenuationPropertyValues<
    specfem::constants::N_SLS>
    specfem::attenuation::get_attenuation_property_values<
        specfem::constants::N_SLS>(
        Kokkos::View<type_real[specfem::constants::N_SLS], Kokkos::LayoutRight,
                     Kokkos::HostSpace>,
        Kokkos::View<type_real[specfem::constants::N_SLS], Kokkos::LayoutRight,
                     Kokkos::HostSpace>);

template type_real specfem::attenuation::get_attenuation_scale_factor<
    specfem::constants::N_SLS>(
    type_real,
    Kokkos::View<type_real[specfem::constants::N_SLS], Kokkos::LayoutRight,
                 Kokkos::HostSpace>,
    Kokkos::View<type_real[specfem::constants::N_SLS], Kokkos::LayoutRight,
                 Kokkos::HostSpace>,
    type_real, type_real);
