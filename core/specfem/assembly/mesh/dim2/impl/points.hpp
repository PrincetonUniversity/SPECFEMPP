#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief Spectral element assembly information
 *
 * This struct contains information about the coordinates and their local
 * to global mapping.
 *
 */
template <> struct points<specfem::dimension::type::dim2> {
public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension
  constexpr static int ndim = 2;      ///< Number of dimensions
  int nspec;                          ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z dimension
  int ngllx; ///< Number of quadrature points in x dimension
  int nglob; ///< Number of global quadrature points

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

  points(const int &nspec, const int &ngllz, const int &ngllx, const int &nglob)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx), nglob(nglob),
        index_mapping("specfem::assembly::points::index_mapping", nspec, ngllz,
                      ngllx),
        coord("specfem::assembly::points::coord", ndim, nspec, ngllz, ngllx),
        h_index_mapping(Kokkos::create_mirror_view(index_mapping)),
        h_coord(Kokkos::create_mirror_view(coord)) {}

  // Constructor that takes pre-computed coordinate arrays
  points(const int &nspec, const int &ngllz, const int &ngllx, const int &nglob,
         IndexMappingViewType::HostMirror h_index_mapping_in,
         CoordViewType::HostMirror h_coord_in, type_real xmin_in,
         type_real xmax_in, type_real zmin_in, type_real zmax_in)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx), nglob(nglob),
        index_mapping("specfem::assembly::points::index_mapping", nspec, ngllz,
                      ngllx),
        coord("specfem::assembly::points::coord", ndim, nspec, ngllz, ngllx),
        h_index_mapping(h_index_mapping_in), h_coord(h_coord_in), xmin(xmin_in),
        xmax(xmax_in), zmin(zmin_in), zmax(zmax_in) {
    // Copy host data to device
    Kokkos::deep_copy(index_mapping, h_index_mapping);
    Kokkos::deep_copy(coord, h_coord);
  }
};

} // namespace specfem::assembly::mesh_impl
