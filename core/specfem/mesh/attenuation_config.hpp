#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::mesh {

struct attenuation_config {
  bool enabled = false;
  specfem::units::Hertz f0;
  specfem::utilities::Band<specfem::units::Hertz> band;
  Kokkos::View<type_real[specfem::constants::N_SLS], Kokkos::LayoutRight,
               Kokkos::HostSpace>
      tau_sigma;
};

} // namespace specfem::mesh
