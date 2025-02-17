#pragma once
#include "enumerations/dimension.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

template <specfem::dimension::type DimensionType> struct partial_derivatives;

template <> struct partial_derivatives<specfem::dimension::type::dim3> {
  constexpr static auto dimension = specfem::dimension::type::dim3;

  using LocalView = Kokkos::View<type_real ****, Kokkos::HostSpace>;

  // Parameters
  int nspec;
  int ngllx;
  int nglly;
  int ngllz;

  LocalView xix;
  LocalView xiy;
  LocalView xiz;
  LocalView etax;
  LocalView etay;
  LocalView etaz;
  LocalView gammax;
  LocalView gammay;
  LocalView gammaz;
  LocalView jacobian;

  // Constructors
  partial_derivatives(){}; // Default constructor

  // Constructor to initialize the coordinates
  partial_derivatives(int nspec, int ngllx, int nglly, int ngllz)
      : nspec(nspec), ngllx(ngllx), nglly(nglly), ngllz(ngllz),
        xix("xix", nspec, ngllx, nglly, ngllz),
        xiy("xiy", nspec, ngllx, nglly, ngllz),
        xiz("xiz", nspec, ngllx, nglly, ngllz),
        etax("etax", nspec, ngllx, nglly, ngllz),
        etay("etay", nspec, ngllx, nglly, ngllz),
        etaz("etaz", nspec, ngllx, nglly, ngllz),
        gammax("gammax", nspec, ngllx, nglly, ngllz),
        gammay("gammay", nspec, ngllx, nglly, ngllz),
        gammaz("gammaz", nspec, ngllx, nglly, ngllz),
        jacobian("jacobian", nspec, ngllx, nglly, ngllz){};

  void print() const;

  void print(int ispec, int igllx, int iglly, int igllz) const;
};

} // namespace mesh
} // namespace specfem
