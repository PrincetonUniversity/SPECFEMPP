#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/compute.hpp"

namespace specfem {
namespace kokkos_kernels {
namespace impl {

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
void compute_mass_matrix(const type_real &dt,
                         const specfem::compute::assembly &assembly);
} // namespace impl
} // namespace kokkos_kernels
} // namespace specfem
