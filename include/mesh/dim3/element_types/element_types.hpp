#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace mesh {

template <> struct element_types<specfem::dimension::type::dim3> {

  constexpr static auto dimension = specfem::dimension::type::dim3;

  int nspec; ///< Number of spectral elements

  int nelastic;     ///< Number of elastic spectral elements
  int nacoustic;    ///< Number of acoustic spectral elements
  int nporoelastic; ///< Number of poroelastic spectral elements

  template <typename T> using View1D = Kokkos::View<T *, Kokkos::HostSpace>;

  element_types() = default;

  element_types(const int nspec)
      : nspec(nspec), ispec_is_elastic("ispec_is_elastic", nspec),
        ispec_is_acoustic("ispec_is_acoustic", nspec),
        ispec_is_poroelastic("ispec_is_poroelastic", nspec){};

  void set_elements();

  template <specfem::element::medium_tag MediumTag> View1D<int> get_elements();

  void print() const;

  void print(const int ispec) const;

  template <specfem::element::medium_tag MediumTag>
  void print(const int i) const;

  View1D<bool> ispec_is_elastic;     ///< Elastic spectral elements
  View1D<bool> ispec_is_acoustic;    ///< Acoustic spectral elements
  View1D<bool> ispec_is_poroelastic; ///< Poroelastic spectral elements

private:
  View1D<int> ispec_elastic;
  View1D<int> ispec_acoustic;
  View1D<int> ispec_poroelastic;
};
} // namespace mesh
} // namespace specfem
