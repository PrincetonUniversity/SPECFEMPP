#ifndef _SPECFEM_KERNELS_IMPL_INTERFACE_KERNELS_HPP
#define _SPECFEM_KERNELS_IMPL_INTERFACE_KERNELS_HPP

#include "coupled_interface/coupled_interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace kokkos_kernels {
namespace impl {
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class interface_kernels;

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionTag>
class interface_kernels<WavefieldType, DimensionTag,
                        specfem::element::medium_tag::elastic_psv> {
public:
  interface_kernels(const specfem::assembly::assembly &assembly)
      : elastic_acoustic_interface(assembly) {}

  inline void compute_coupling() {
    elastic_acoustic_interface.compute_coupling();
  }

private:
  specfem::coupled_interface::coupled_interface<
      WavefieldType, DimensionTag, specfem::element::medium_tag::elastic_psv,
      specfem::element::medium_tag::acoustic>
      elastic_acoustic_interface;
};

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionTag>
class interface_kernels<WavefieldType, DimensionTag,
                        specfem::element::medium_tag::acoustic> {
public:
  interface_kernels(const specfem::assembly::assembly &assembly)
      : acoustic_elastic_interface(assembly) {}

  inline void compute_coupling() {
    acoustic_elastic_interface.compute_coupling();
  }

private:
  specfem::coupled_interface::coupled_interface<
      WavefieldType, DimensionTag, specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::elastic_psv>
      acoustic_elastic_interface;
};

} // namespace impl
} // namespace kokkos_kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_IMPL_INTERFACE_KERNELS_HPP */
