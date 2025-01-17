#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"

namespace specfem {
namespace kokkos_kernels {
namespace impl {

template <specfem::dimension::type DimensionType,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::element::medium_tag MediumTag>
void invert_mass_matrix(const specfem::compute::assembly &assembly);
}

} // namespace kokkos_kernels
} // namespace specfem
