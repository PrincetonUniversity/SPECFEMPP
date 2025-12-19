#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief Template specialization for 2D quadrature points in spectral element
 * mesh
 *
 * This class stores and manages quadrature points for a 2D spectral element
 * mesh, including their coordinates, global indexing, and boundary information.
 * It uses Kokkos views for efficient memory management and device/host data
 * transfers.
 *
 * @tparam specfem::dimension::type::dim2 Template specialization for 2D case
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

  /**
   * @brief Kokkos view for storing global element number for every quadrature
   * point
   */
  using IndexMappingViewType =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view for storing coordinates of every distinct quadrature
   * point
   */
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

  /**
   * @brief Default constructor
   *
   * Creates an uninitialized points object with default values.
   */

  points() = default;
  /**
   * @brief Constructor with mesh parameters
   *
   * Initializes the points object with the given mesh dimensions and allocates
   * Kokkos views for storing quadrature point data.
   *
   * @param nspec Number of spectral elements in the mesh
   * @param ngllz Number of quadrature points in the z (vertical) dimension
   * @param ngllx Number of quadrature points in the x (horizontal) dimension
   * @param nglob Total number of global quadrature points
   */
  points(const int &nspec, const int &ngllz, const int &ngllx, const int &nglob)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx), nglob(nglob),
        index_mapping("specfem::assembly::points::index_mapping", nspec, ngllz,
                      ngllx),
        coord("specfem::assembly::points::coord", ndim, nspec, ngllz, ngllx),
        h_index_mapping(Kokkos::create_mirror_view(index_mapping)),
        h_coord(Kokkos::create_mirror_view(coord)) {}

  /**
   * @brief Constructor with pre-computed coordinate data
   *
   * Initializes the points object with pre-computed coordinate arrays and
   * boundary information. This constructor copies the provided host data
   * to device memory using Kokkos deep_copy operations.
   *
   * @param nspec Number of spectral elements in the mesh
   * @param ngllz Number of quadrature points in the z dimension
   * @param ngllx Number of quadrature points in the x dimension
   * @param nglob Total number of global quadrature points
   * @param h_index_mapping_in Pre-computed host mirror of index mapping
   * @param h_coord_in Pre-computed host mirror of coordinates
   * @param xmin_in Minimum x coordinate value
   * @param xmax_in Maximum x coordinate value
   * @param zmin_in Minimum z coordinate value
   * @param zmax_in Maximum z coordinate value
   */
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
