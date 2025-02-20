#include "mesh/dim3/materials/materials.hpp"
#include "enumerations/dimension.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <limits>

void specfem::mesh::materials<specfem::dimension::type::dim3>::print() {

  std::cout << "---------------------------------" << std::endl;
  std::cout << "Materials: " << std::endl;
  std::cout << "---------------------------------" << std::endl;
  std::cout << "nspec: " << nspec << std::endl;
  std::cout << "ngllx: " << ngllx << std::endl;
  std::cout << "nglly: " << nglly << std::endl;
  std::cout << "ngllz: " << ngllz << std::endl;
  std::cout << "acoustic: " << acoustic << std::endl;
  std::cout << "elastic: " << elastic << std::endl;
  std::cout << "poroelastic: " << poroelastic << std::endl;
  std::cout << "anisotropic: " << anisotropic << std::endl;

  // Initialize min and max with appropriate extreme values
  type_real kappa_min = std::numeric_limits<type_real>::max();
  type_real kappa_max = std::numeric_limits<type_real>::lowest();

  Kokkos::parallel_reduce(
      "kappa_min_max", nspec * ngllx * nglly * ngllz,
      KOKKOS_LAMBDA(const int &i, type_real &lmin, type_real &lmax) {
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

  std::cout << "Kappa min/max: " << kappa_min << "/" << kappa_max << std::endl;

  // Initialize min and max with appropriate extreme values
  type_real mu_min = std::numeric_limits<type_real>::max();
  type_real mu_max = std::numeric_limits<type_real>::lowest();

  Kokkos::parallel_reduce(
      "mu_min_max", nspec * ngllx * nglly * ngllz,
      KOKKOS_LAMBDA(const int &i, type_real &lmin, type_real &lmax) {
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

  std::cout << "Mu min/max: " << mu_min << "/" << mu_max << std::endl;

  std::cout << "---------------------------------" << std::endl;
}
