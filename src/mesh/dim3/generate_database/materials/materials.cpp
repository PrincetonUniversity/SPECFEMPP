#include "mesh/dim3/generate_database/materials/materials.hpp"
#include "enumerations/dimension.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <limits>
#include <sstream>

std::string specfem::mesh::materials<specfem::dimension::type::dim3>::print() {

  std::ostringstream message;
  message << "---------------------------------"
          << "\n";
  message << "Materials: "
          << "\n";
  message << "---------------------------------"
          << "\n";
  message << "nspec: " << nspec << "\n";
  message << "ngllx: " << ngllx << "\n";
  message << "nglly: " << nglly << "\n";
  message << "ngllz: " << ngllz << "\n";
  message << "acoustic: " << acoustic << "\n";
  message << "elastic: " << elastic << "\n";
  message << "poroelastic: " << poroelastic << "\n";
  message << "anisotropic: " << anisotropic << "\n";

  // Initialize min and max with appropriate extreme values
  type_real kappa_min = std::numeric_limits<type_real>::max();
  type_real kappa_max = std::numeric_limits<type_real>::lowest();

  Kokkos::parallel_reduce(
      "kappa_min_max",
      specfem::kokkos::HostRange(0, nspec * ngllx * nglly * ngllz),
      KOKKOS_CLASS_LAMBDA(const int &i, type_real &lmin, type_real &lmax) {
        // Compute 3D indices from 1D index i
        const int ispec = i / (ngllx * nglly * ngllz);
        const int j = (i / (ngllx * ngllz)) % nglly;
        const int k = (i / ngllx) % ngllz;
        const int l = i % ngllx;

        const type_real val = kappa(ispec, j, k, l);

        // Update local min and max
        lmin = val < lmin ? val : lmin;
        lmax = val > lmax ? val : lmax;
      },
      Kokkos::Min<type_real>(kappa_min), Kokkos::Max<type_real>(kappa_max));

  message << "Kappa min/max: " << kappa_min << "/" << kappa_max << "\n";

  // Initialize min and max with appropriate extreme values
  type_real mu_min = std::numeric_limits<type_real>::max();
  type_real mu_max = std::numeric_limits<type_real>::lowest();

  Kokkos::parallel_reduce(
      "mu_min_max",
      specfem::kokkos::HostRange(0, nspec * ngllx * nglly * ngllz),
      KOKKOS_CLASS_LAMBDA(const int &i, type_real &lmin, type_real &lmax) {
        // Compute 3D indices from 1D index i
        const int ispec = i / (ngllx * nglly * ngllz);
        const int j = (i / (ngllx * ngllz)) % nglly;
        const int k = (i / ngllx) % ngllz;
        const int l = i % ngllx;

        const type_real val = mu(ispec, j, k, l);

        // Update local min and max
        lmin = val < lmin ? val : lmin;
        lmax = val > lmax ? val : lmax;
      },
      Kokkos::Min<type_real>(mu_min), Kokkos::Max<type_real>(mu_max));

  message << "Mu min/max: " << mu_min << "/" << mu_max << "\n";

  message << "---------------------------------"
          << "\n";

  return message.str();
}
