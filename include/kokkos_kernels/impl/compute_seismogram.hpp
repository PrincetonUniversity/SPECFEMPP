#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"

namespace specfem {
namespace kokkos_kernels {
namespace impl {

template <specfem::dimension::type DimensionType,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void compute_seismograms(specfem::compute::assembly &assembly,
                         const int &isig_step);

} // namespace impl
} // namespace kokkos_kernels
} // namespace specfem
