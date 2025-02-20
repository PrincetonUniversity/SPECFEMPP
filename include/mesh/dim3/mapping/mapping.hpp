#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

template <specfem::dimension::type DimensionType> struct mapping;

template <> struct mapping<specfem::dimension::type::dim3> {

  constexpr static auto dimension = specfem::dimension::type::dim3;

  using UniqueViewInt = Kokkos::View<int *, Kokkos::HostSpace>;
  using UniqueViewBool = Kokkos::View<bool *, Kokkos::HostSpace>;
  using LocalViewInt = Kokkos::View<int ****, Kokkos::HostSpace>;

  // Parameters needed for ibool mapping
  int nspec;
  int nglob;
  int nspec_irregular;

  int ngllx;
  int nglly;
  int ngllz;

  // I do not know currently what these are used for
  type_real xix_regular;
  type_real jacobian_regular;

  // Indices of irregular elements size nspec_irregular
  UniqueViewInt irregular_elements;

  // ibool size nspec, ngllx, nglly, ngllz
  LocalViewInt ibool;

  // Boolean array size nspec
  UniqueViewBool is_acoustic;
  UniqueViewBool is_elastic;
  UniqueViewBool is_poroelastic;

  // Constructors
  mapping(){}; // Default constructor

  // Constructor to initialize the mapping
  mapping(int nspec, int nglob, int nspec_irregular, int ngllx, int nglly,
          int ngllz)
      : nspec(nspec), nglob(nglob), nspec_irregular(nspec_irregular),
        ngllx(ngllx), nglly(nglly), ngllz(ngllz), xix_regular(0.0),
        jacobian_regular(0.0),
        irregular_elements("irregular_elements", nspec_irregular),
        ibool("ibool", nspec, ngllx, nglly, ngllz),
        is_acoustic("is_acoustic", nspec), is_elastic("is_elastic", nspec),
        is_poroelastic("is_poroelastic", nspec){};

  void print() const;

  void print(const int ispec) const;

  template <specfem::element::medium_tag MediumTag>
  void print(const int i) const;
};

} // namespace mesh
} // namespace specfem
