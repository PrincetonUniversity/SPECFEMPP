#pragma once

#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace mesh {

template <> struct mass_matrix<specfem::dimension::type::dim3> {

  constexpr static auto dimension = specfem::dimension::type::dim3;

  int nglob;
  bool is_acoustic;
  bool is_elastic;
  bool is_poroelastic;
  bool has_ocean_load;

  template <typename T> using View1D = Kokkos::View<T *, Kokkos::HostSpace>;

  mass_matrix() = default;

  mass_matrix(const int nglob, const bool is_acoustic, const bool is_elastic,
              const bool is_poroelastic, const bool has_ocean_load)
      : nglob(nglob), is_acoustic(is_acoustic), is_elastic(is_elastic),
        is_poroelastic(is_poroelastic), has_ocean_load(has_ocean_load) {
    // Only allocate the mass matrix if the simulation includes the respective
    // physics.
    if (is_acoustic) {
      acoustic =
          View1D<type_real>("specfem::mesh::mass_matrix::acoustic", nglob);
    }
    if (is_elastic) {
      elastic = View1D<type_real>("specfem::mesh::mass_matrix::elastic", nglob);
      if (has_ocean_load) {
        ocean_load =
            View1D<type_real>("specfem::mesh::mass_matrix::ocean_load", nglob);
      }
    }
    if (is_poroelastic) {
      solid_poroelastic = View1D<type_real>(
          "specfem::mesh::mass_matrix::solid_poroelastic", nglob);
      fluid_poroelastic = View1D<type_real>(
          "specfem::mesh::mass_matrix::fluid_poroelastic", nglob);
    }
  };

  View1D<type_real> elastic;
  View1D<type_real> acoustic;
  View1D<type_real> ocean_load;
  View1D<type_real> solid_poroelastic;
  View1D<type_real> fluid_poroelastic;
};

} // namespace mesh
} // namespace specfem
