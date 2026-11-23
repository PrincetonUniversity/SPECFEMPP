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
    const Kokkos::View<type_real[specfem::constants::N_SLS],
                       Kokkos::LayoutRight, Kokkos::HostSpace> &,
    const Kokkos::View<type_real[specfem::constants::N_SLS],
                       Kokkos::LayoutRight, Kokkos::HostSpace> &);

template specfem::attenuation::ModulusScales
specfem::attenuation::get_attenuation_modulus_scales<specfem::constants::N_SLS>(
    type_real,
    const Kokkos::View<type_real[specfem::constants::N_SLS],
                       Kokkos::LayoutRight, Kokkos::HostSpace> &,
    const Kokkos::View<type_real[specfem::constants::N_SLS],
                       Kokkos::LayoutRight, Kokkos::HostSpace> &,
    type_real, type_real);
