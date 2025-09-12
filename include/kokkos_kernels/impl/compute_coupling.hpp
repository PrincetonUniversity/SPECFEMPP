#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"
#include <type_traits>

namespace specfem::kokkos_kernels::impl {

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
void compute_coupling(
    std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::weakly_conforming> /*unused*/,
    const specfem::assembly::assembly<DimensionTag> &assembly);

template <specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          typename... AdditionalArgs>
void compute_coupling(
    const specfem::assembly::assembly<DimensionTag> &assembly) {
  using connection_dispatch =
      std::integral_constant<specfem::connections::type, ConnectionTag>;

  compute_coupling<DimensionTag, WavefieldType, InterfaceTag, BoundaryTag>(
      connection_dispatch(), assembly);
}
} // namespace specfem::kokkos_kernels::impl
