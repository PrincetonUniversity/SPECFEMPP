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
  /// Attenuate the bulk modulus as well as the shear modulus. Globe meshes
  /// set this to false: SPECFEM3D_GLOBE implements Q_mu attenuation only
  /// (compute_element_att_memory.F90) and shifts only the shear moduli to
  /// unrelaxed (prepare_attenuation.f90), so kappa is left physical and
  /// carries no memory variables.
  bool bulk_attenuation = true;
};

} // namespace specfem::mesh
