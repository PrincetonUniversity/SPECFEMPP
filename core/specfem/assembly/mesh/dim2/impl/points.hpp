#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief Spectral element assembly information
 *
 */
template <> struct points<specfem::dimension::type::dim2> {
public:
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension
  int nspec;                          ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z dimension
  int ngllx; ///< Number of quadrature points in x dimension

  using IndexMappingViewType =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  using CoordViewType = Kokkos::View<type_real ****, Kokkos::LayoutRight,
                                     Kokkos::DefaultExecutionSpace>;

  IndexMappingViewType index_mapping; ///< Global index
                                      ///< number for every
                                      ///< quadrature point
  CoordViewType coord;                ///< (x, z) for every distinct
                                      ///< quadrature point
  IndexMappingViewType::HostMirror h_index_mapping; ///< Global element
                                                    ///< number for every
                                                    ///< quadrature point
  CoordViewType::HostMirror h_coord;                ///< (x, z) for every
                                                    ///< distinct quadrature
                                                    ///< point
  type_real xmin, xmax, zmin, zmax; ///< Min and max values of x and z
                                    ///< coordinates

  points() = default;

  points(const int &nspec, const int &ngllz, const int &ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        index_mapping("specfem::assembly::points::index_mapping", nspec, ngllz,
                      ngllx),
        coord("specfem::assembly::points::coord", ndim, nspec, ngllz, ngllx),
        h_index_mapping(Kokkos::create_mirror_view(index_mapping)),
        h_coord(Kokkos::create_mirror_view(coord)) {}
};

} // namespace specfem::assembly::mesh_impl
