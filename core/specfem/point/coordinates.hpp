#pragma once

#include "enumerations/dimension.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace specfem {
namespace point {

/**
 * @brief Local reference coordinates for quadrature points within spectral
 * elements.
 *
 * Local coordinates represent the position of quadrature points within the
 * reference element coordinate system, typically ranging from -1 to +1 in each
 * spatial direction. These coordinates are essential for the spectral element
 * method as they define the locations where basis functions are evaluated and
 * where numerical integration is performed.
 *
 * @tparam DimensionTag Spatial dimension of the element (dim2 for 2D, dim3 for
 * 3D). Determines the number of coordinate components stored.
 *
 * @note Local coordinates typically use the reference domain [-1, +1]^d where d
 * is the spatial dimension, following standard spectral element conventions.
 *
 * @see specfem::point::global_coordinates for physical space coordinates
 * @see specfem::point::jacobian_matrix for coordinate transformation
 *
 * @code
 * // Example: 2D local coordinates at element center
 * specfem::point::local_coordinates<specfem::dimension::type::dim2>
 * local_coords; local_coords.ispec = element_id; local_coords.xi = 0.0;     //
 * Center in xi direction local_coords.gamma = 0.0;  // Center in gamma
 * direction
 *
 * // Transform to global coordinates
 * auto global_coords = transform_to_global(local_coords, jacobian);
 * @endcode
 */
template <specfem::dimension::type DimensionTag> struct local_coordinates;

/**
 * @brief Global physical coordinates for quadrature points in the computational
 * domain.
 *
 * Global coordinates represent the actual physical positions of quadrature
 * points within the computational domain. These coordinates correspond to
 * real-world locations and are obtained by applying isoparametric
 * transformations from the local reference coordinates using the element
 * geometry and Jacobian information.
 *
 * Global coordinates are essential for:
 * - Applying boundary conditions at specific physical locations
 * - Computing distances between points for source-receiver calculations
 * - Evaluating spatially-varying material properties
 * - Post-processing and visualization of simulation results
 * - Applying external forces and source terms
 *
 * The transformation from local to global coordinates is given by:
 * \f$
 *   \mathbf{x} = \sum_{i=1}^{N} \mathbf{x}_i \phi_i(\boldsymbol{\xi})
 * \f$
 * where \f$\mathbf{x}_i\f$ are the element node coordinates, \f$\phi_i\f$ are
 * the basis functions, and \f$\boldsymbol{\xi}\f$ are the local coordinates.
 *
 * @tparam DimensionTag Spatial dimension of the problem (dim2 for 2D, dim3 for
 * 3D). Determines the coordinate space dimensionality.
 *
 * @note Global coordinates use the physical units of the problem (meters,
 * kilometers) and represent actual positions in the computational domain.
 *
 * @see specfem::point::local_coordinates for reference element coordinates
 * @see specfem::point::distance for computing distances between points
 *
 * @code
 * // Example: Computing distance between two points
 * specfem::point::global_coordinates<specfem::dimension::type::dim2> p1, p2;
 * p1.x = 100.0; p1.z = 50.0;   // Point 1 in meters
 * p2.x = 200.0; p2.z = 150.0;  // Point 2 in meters
 *
 * type_real dist = specfem::point::distance(p1, p2);
 * // dist ≈ 141.42 meters
 * @endcode
 */
template <specfem::dimension::type DimensionTag> struct global_coordinates;

/**
 * @brief Compute Euclidean distance between two points in global coordinates.
 *
 * Calculates the straight-line distance between two quadrature points specified
 * in global physical coordinates. This function is commonly used in:
 * - Source-receiver distance calculations for seismic simulations
 * - Spatial correlation computations
 * - Mesh quality assessment
 * - Post-processing distance-based analyses
 *
 * The distance is computed using the standard Euclidean norm:
 * \f$
 *   d = \sqrt{\sum_{i=1}^{d} (x_{2,i} - x_{1,i})^2}
 * \f$
 * where \f$d\f$ is the spatial dimension and \f$x_{j,i}\f$ are the coordinate
 * components.
 *
 * @tparam DimensionTag Spatial dimension determining coordinate space (dim2 or
 * dim3).
 * @param p1 First point in global coordinates.
 * @param p2 Second point in global coordinates.
 * @return type_real Euclidean distance between the two points in physical
 * units.
 *
 * @note The function is marked with `KOKKOS_FUNCTION` for device/host
 * portability and can be used within parallel kernels.
 *
 * @see specfem::point::global_coordinates for coordinate definitions
 *
 * @code
 * // Example: Computing source-receiver distance
 * specfem::point::global_coordinates<specfem::dimension::type::dim2> source,
 * receiver; source.x = 0.0; source.z = 0.0;        // Source at origin
 * receiver.x = 1000.0; receiver.z = 500.0; // Receiver 1 km away
 *
 * type_real distance = specfem::point::distance(source, receiver);
 * // distance ≈ 1118.03 meters
 *
 * // Use in travel time calculation
 * type_real travel_time = distance / wave_velocity;
 * @endcode
 */
template <specfem::dimension::type DimensionTag>
KOKKOS_FUNCTION type_real
distance(const specfem::point::global_coordinates<DimensionTag> &p1,
         const specfem::point::global_coordinates<DimensionTag> &p2);

//-------------------------- 2D Specializations ------------------------------//

/**
 * @brief 2D specialization for local reference coordinates within spectral
 * elements.
 *
 * This specialization stores local coordinates for 2D spectral elements using
 * the reference coordinate system with ξ (xi) and γ (gamma) directions. The
 * reference element is typically the standard square [-1, +1]² in the ξ-γ
 * coordinate space.
 *
 * The local coordinate system follows these conventions:
 * - ξ (xi): Horizontal direction in reference space, typically mapped to
 * x-direction
 * - γ (gamma): Vertical direction in reference space, typically mapped to
 * z-direction
 * - Both coordinates range from -1 to +1 for the standard reference element
 *
 * Local coordinates are fundamental for:
 * - Evaluating 2D Lagrange polynomial basis functions
 * - Computing basis function derivatives: ∂φ/∂ξ, ∂φ/∂γ
 * - Performing 2D GLL quadrature integration
 * - Interpolating field values within 2D elements
 *
 * @note The element index (ispec) links these local coordinates to a specific
 *       element in the mesh, enabling the transformation to global coordinates.
 *
 * @see specfem::quadrature::gll for 2D quadrature point locations
 * @see specfem::basis for 2D basis function evaluation
 *
 * @code
 * // Example: GLL quadrature point at (ξ=0.5, γ=-0.5)
 * specfem::point::local_coordinates<specfem::dimension::type::dim2> local_pt;
 * local_pt.ispec = element_index;
 * local_pt.xi = 0.5;     // Offset to the right in reference element
 * local_pt.gamma = -0.5; // Offset downward in reference element
 *
 * // Use in basis function evaluation
 * auto basis_values = evaluate_basis_functions(local_pt.xi, local_pt.gamma);
 * @endcode
 */
template <> struct local_coordinates<specfem::dimension::type::dim2> {
  int ispec; ///< Spectral element index identifying which element contains
             ///< this quadrature point. Used for element-specific operations
             ///< and coordinate transformations to global space.

  type_real xi; ///< Local coordinate \f$ \xi \f$ in the horizontal reference
                ///< direction. Ranges from -1 to +1 in the standard reference
                ///< element, corresponding to the left-to-right element extent.

  type_real gamma; ///< Local coordinate \f$ \gamma \f$ in the vertical
                   ///< reference direction. Ranges from -1 to +1 in the
                   ///< standard reference element, corresponding to the
                   ///< bottom-to-top element extent.

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  local_coordinates() = default;

  /**
   * @brief Construct a new local coordinates object
   *
   * @param ispec Index of the spectral element
   * @param xi Local coordinate \f$ \xi \f$
   * @param gamma Local coordinate \f$ \gamma \f$
   */
  KOKKOS_FUNCTION
  local_coordinates(const int &ispec, const type_real &xi,
                    const type_real &gamma)
      : ispec(ispec), xi(xi), gamma(gamma) {}

  /**
   * @brief Construct a new local coordinates object from element index and
   * Kokkos array
   *
   * @param ispec Index of the spectral element
   * @param coords Kokkos 1D array containing [xi, gamma] coordinates
   */
  template <typename ViewType>
  KOKKOS_FUNCTION local_coordinates(const int &ispec, const ViewType &coords)
      : ispec(ispec), xi(coords[0]), gamma(coords[1]) {
    static_assert(ViewType::rank() == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 2,
                  "ViewType must have extent 2 for 2D coordinates");
  }
};

/**
 * @brief Template specialization for 2D elements
 *
 */
template <> struct global_coordinates<specfem::dimension::type::dim2> {
  type_real x; ///< Global coordinate \f$ x \f$
  type_real z; ///< Global coordinate \f$ z \f$

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  global_coordinates() = default;

  /**
   * @brief Construct a new global coordinates object
   *
   * @param x Global coordinate \f$ x \f$
   * @param z Global coordinate \f$ z \f$
   */
  KOKKOS_FUNCTION
  global_coordinates(const type_real &x, const type_real &z) : x(x), z(z) {}

  /**
   * @brief Construct a new global coordinates object from Kokkos array
   *
   * @param coords Kokkos 1D array containing [x, z] coordinates
   */
  template <typename ViewType>
  KOKKOS_FUNCTION global_coordinates(const ViewType &coords)
      : x(coords[0]), z(coords[1]) {
    static_assert(ViewType::rank() == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 2,
                  "ViewType must have extent 2 for 2D coordinates");
  }
};

//-------------------------- 3D Specializations ------------------------------//

/**
 * @brief Template specialization for 3D elements
 *
 */
template <> struct local_coordinates<specfem::dimension::type::dim3> {
  int ispec;       ///< Index of the spectral element
  type_real xi;    ///< Local coordinate \f$ \xi \f$
  type_real eta;   ///< Local coordinate \f$ \eta \f$
  type_real gamma; ///< Local coordinate \f$ \gamma \f$

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  local_coordinates() = default;

  /**
   * @brief Construct a new local coordinates object
   *
   * @param ispec Index of the spectral element
   * @param xi Local coordinate \f$ \xi \f$
   * @param eta Local coordinate \f$ \eta \f$
   * @param gamma Local coordinate \f$ \gamma \f$
   */
  KOKKOS_FUNCTION
  local_coordinates(const int &ispec, const type_real &xi, const type_real &eta,
                    const type_real &gamma)
      : ispec(ispec), xi(xi), eta(eta), gamma(gamma) {}

  /**
   * @brief Construct a new local coordinates object from element index and
   * Kokkos array
   *
   * @param ispec Index of the spectral element
   * @param coords Kokkos 1D array containing [xi, eta, gamma] coordinates
   */
  template <typename ViewType>
  KOKKOS_FUNCTION local_coordinates(const int &ispec, const ViewType &coords)
      : ispec(ispec), xi(coords[0]), eta(coords[1]), gamma(coords[2]) {
    static_assert(ViewType::rank() == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 3,
                  "ViewType must have extent 3 for 3D coordinates");
  }
};

/**
 * @brief Template specialization for 3D elements
 *
 */
template <> struct global_coordinates<specfem::dimension::type::dim3> {
  type_real x; ///< Global coordinate \f$ x \f$
  type_real y; ///< Global coordinate \f$ y \f$
  type_real z; ///< Global coordinate \f$ z \f$

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  global_coordinates() = default;

  /**
   * @brief Construct a new global coordinates object
   *
   * @param x Global coordinate \f$ x \f$
   * @param y Global coordinate \f$ y \f$
   * @param z Global coordinate \f$ z \f$
   */
  KOKKOS_FUNCTION
  global_coordinates(const type_real &x, const type_real &y, const type_real &z)
      : x(x), y(y), z(z) {}

  /**
   * @brief Construct a new global coordinates object from Kokkos array
   *
   * @param coords Kokkos 1D array containing [x, y, z] coordinates
   */
  template <typename ViewType>
  KOKKOS_FUNCTION global_coordinates(const ViewType &coords)
      : x(coords[0]), y(coords[1]), z(coords[2]) {
    static_assert(ViewType::rank() == 1, "ViewType must be rank 1");
    static_assert(ViewType::static_extent(0) == 3,
                  "ViewType must have extent 3 for 3D coordinates");
  }
};

} // namespace point
} // namespace specfem

template <specfem::dimension::type Dimension>
std::ostream &
operator<<(std::ostream &s,
           const specfem::point::global_coordinates<Dimension> &point);
