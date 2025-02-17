#pragma once
#include "enumerations/dimension.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

template <specfem::dimension::type DimensionType> struct coordinates;

template <> struct coordinates<specfem::dimension::type::dim3> {
  constexpr static auto dimension = specfem::dimension::type::dim3;

  using UniqueView = Kokkos::View<type_real *, Kokkos::HostSpace>;
  using LocalView = Kokkos::View<type_real ****, Kokkos::HostSpace>;

  // Parameters
  int nspec;
  int nglob;
  int ngllx;
  int nglly;
  int ngllz;

  UniqueView x;
  UniqueView y;
  UniqueView z;

  // Constructors
  coordinates(){}; // Default constructor

  // Constructor to initialize the coordinates
  coordinates(int nspec, int nglob, int ngllx, int nglly, int ngllz)
      : nspec(nspec), nglob(nglob), ngllx(ngllx), nglly(nglly), ngllz(ngllz),
        x("x", nglob), y("y", nglob), z("z", nglob){};

  void print() const;

  void print(int iglob) const;

  void print(int ispec, int igllx, int iglly, int igllz) const;
};

} // namespace mesh
} // namespace specfem
