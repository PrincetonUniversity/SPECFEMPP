#pragma once

#include "datatypes/point_view.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {
/**
 * @brief Template class for storing boundary condition data at spectral element
 * quadrature points.
 *
 * The boundary class is a template-based data structure that encapsulates
 * boundary condition information for quadrature points within spectral elements
 * in SPECFEMPP finite element simulations. It supports various boundary types
 * including acoustic free surface, Stacey absorbing boundaries, and composite
 * boundary conditions.
 *
 * @tparam BoundaryTag Enumerated type indicating the specific boundary
 * condition type. Valid values include:
 *                     - `specfem::element::boundary_tag::none`: No boundary
 * condition
 *                     -
 * `specfem::element::boundary_tag::acoustic_free_surface`: Free surface
 * condition
 *                     - `specfem::element::boundary_tag::stacey`: Stacey
 * absorbing boundary
 *                     -
 * `specfem::element::boundary_tag::composite_stacey_dirichlet`: Combined
 * boundary
 *
 * @tparam DimensionTag Spatial dimension of the spectral element. Typically
 * `specfem::dimension::type::dim2` or `specfem::dimension::type::dim3` for
 * 2D/3D simulations respectively.
 *
 * @tparam UseSIMD Boolean template parameter controlling SIMD vectorization
 * support. When `true`, enables SIMD operations for performance optimization.
 *                 When `false`, uses scalar operations.
 *
 * @see specfem::element::boundary_tag
 * @see specfem::data_access::Accessor
 * @see specfem::datatype::simd
 *
 * @note This is a primary template declaration. Actual functionality is
 * provided through explicit template specializations for each boundary
 * condition type.
 *
 * @since Added with pitchfork refactoring (commit f8b6d1b1)
 *
 * @code
 * // Example: Creating boundary conditions for 2D acoustic simulation
 * using AcousticBoundary = specfem::point::boundary<
 *     specfem::element::boundary_tag::acoustic_free_surface,
 *     specfem::dimension::type::dim2, false>;
 *
 * AcousticBoundary boundary;
 * boundary.tag += specfem::element::boundary_tag::acoustic_free_surface;
 *
 * // SIMD-enabled Stacey boundary
 * using StaceySIMD = specfem::point::boundary<
 *     specfem::element::boundary_tag::stacey,
 *     specfem::dimension::type::dim2, true>;
 * @endcode
 */
template <specfem::element::boundary_tag BoundaryTag,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary;

/**
 * @brief Template specialization for interior quadrature points with no
 * boundary conditions.
 *
 * This specialization represents quadrature points that are not located on any
 * boundary surface of the computational domain. It serves as the base class for
 * all other boundary condition types, providing the fundamental data storage
 * and access patterns.
 *
 * The class inherits from the SPECFEMPP data access system to enable efficient
 * memory management and device/host data transfers in heterogeneous computing
 * environments.
 *
 * @tparam DimensionTag Spatial dimension of the spectral element where the
 * quadrature point is located. Must be a valid `specfem::dimension::type`
 * value.
 * @tparam UseSIMD Boolean flag controlling SIMD vectorization. When `true`,
 * enables vectorized operations on multiple quadrature points simultaneously.
 *
 * @note This specialization forms the root of the inheritance hierarchy for all
 * boundary types. Other boundary conditions (acoustic free surface, Stacey,
 * composite) inherit from this or from each other to build layered
 * functionality.
 *
 * @see specfem::data_access::Accessor
 * @see specfem::element::boundary_tag_container
 *
 * @code
 * // Example: Basic interior point boundary
 * using InteriorBoundary = specfem::point::boundary<
 *     specfem::element::boundary_tag::none,
 *     specfem::dimension::type::dim2, false>;
 *
 * InteriorBoundary interior;
 * // Tag remains unset for interior points
 * assert(interior.tag.get_tag() == specfem::element::boundary_tag::none);
 * @endcode
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::none, DimensionTag, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::boundary, DimensionTag,
          UseSIMD> {
private:
  // We use simd_like vector to store tags. Tags are stored as enums, so a simd
  // type is ill-defined for them. However, we use scalar array types of size
  // simd<type_real>::size() to store them. The goal of this approach is to use
  // tags to mask a type_real simd vector and perform SIMD operations on those
  // SIMD vectors.
  using value_type = typename specfem::datatype::simd_like<
      specfem::element::boundary_tag_container, type_real,
      UseSIMD>::datatype; ///< Datatype for storing values. Is a scalar if
                          ///< UseSIMD is false, otherwise is a SIMD like
                          ///< vector.

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD data type
  ///@}

  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::none; ///< Tag indicating no boundary
                                            ///< condition
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor for interior boundary points.
   *
   * Initializes a boundary object representing an interior quadrature point
   * with no specific boundary conditions. The boundary tag is left
   * uninitialized and should be set explicitly if needed.
   *
   * @note This constructor is marked with `KOKKOS_FUNCTION` to enable execution
   *       on both CPU and GPU devices in Kokkos parallel regions.
   */
  KOKKOS_FUNCTION
  boundary() = default;
  ///@}

  value_type tag; ///< Tag indicating the type of boundary condition at the
                  ///< quadrature point
};

/**
 * @brief Template specialization for acoustic free surface boundary condition.
 *
 * This specialization handles quadrature points located on free surfaces in
 * acoustic wave propagation problems. Free surface boundaries enforce a
 * Dirichlet condition where the pressure (or displacement in certain
 * formulations) is constrained to zero, representing the interface between the
 * computational domain and free space or vacuum.
 *
 * Mathematically, the free surface condition can be expressed as:
 * \f$
 *   p|_{\Gamma_{\text{free}}} = 0
 * \f$
 * where \f$p\f$ is the acoustic pressure and \f$\Gamma_{\text{free}}\f$ denotes
 * the free surface boundary.
 *
 * The class inherits from the base `none` boundary type and adds specific
 * handling for acoustic free surface physics. It supports conversion from
 * composite boundary types that include free surface components.
 *
 * @tparam DimensionTag Spatial dimension of the spectral element. Typically
 *                      `specfem::dimension::type::dim2` for 2D problems.
 * @tparam UseSIMD Boolean controlling SIMD vectorization for performance
 * optimization.
 *
 * @note Free surface conditions are commonly used to model the top surface of
 *       geological domains (e.g., the Earth's surface in seismic simulations).
 *
 * @see specfem::boundary_conditions::acoustic_free_surface_type
 * @see boundary<specfem::element::boundary_tag::composite_stacey_dirichlet,
 * DimensionTag, UseSIMD>
 *
 * @code
 * // Example: Acoustic free surface boundary setup
 * using FreeSurfaceBoundary = specfem::point::boundary<
 *     specfem::element::boundary_tag::acoustic_free_surface,
 *     specfem::dimension::type::dim2, false>;
 *
 * FreeSurfaceBoundary boundary;
 * boundary.tag += specfem::element::boundary_tag::acoustic_free_surface;
 *
 * // The boundary condition zeroes acceleration components:
 * // acceleration.acceleration(icomp) = 0.0 for all components
 * @endcode
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::acoustic_free_surface,
                DimensionTag, UseSIMD>
    : public boundary<specfem::element::boundary_tag::none, DimensionTag,
                      UseSIMD> {
public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  ///@}
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::acoustic_free_surface;
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor for acoustic free surface boundary.
   *
   * Creates an acoustic free surface boundary object with uninitialized
   * boundary tags. The tag should be explicitly set to identify this as
   * a free surface boundary point.
   *
   * @note Marked with `KOKKOS_FUNCTION` for device/host portability.
   */
  KOKKOS_FUNCTION
  boundary() = default;

  /**
   * @brief Implicit conversion constructor from composite Stacey-Dirichlet
   * boundary.
   *
   * Enables extraction of the acoustic free surface component from a composite
   * boundary condition that combines multiple boundary types. This constructor
   * copies the boundary tag information while discarding Stacey-specific data
   * (edge weights and normals).
   *
   * This conversion is essential in the SPECFEMPP boundary condition processing
   * pipeline, where composite boundaries are decomposed into their constituent
   * parts for specialized handling.
   *
   * @param boundary Reference to a composite Stacey-Dirichlet boundary object
   *                 containing both free surface and absorbing boundary data.
   *
   * @note The conversion preserves only the boundary tag; geometric information
   *       specific to Stacey boundaries (edge normal vectors and integration
   * weights) is not transferred.
   *
   * @see boundary<specfem::element::boundary_tag::composite_stacey_dirichlet,
   * DimensionTag, UseSIMD>
   */
  KOKKOS_FUNCTION
  boundary(const specfem::point::boundary<
           specfem::element::boundary_tag::composite_stacey_dirichlet,
           DimensionTag, UseSIMD> &boundary);
  ///@}
};

/**
 * @brief Template specialization for Stacey absorbing boundary condition.
 *
 * This specialization implements Stacey absorbing boundary conditions, which
 * are widely used in seismic wave propagation simulations to prevent artificial
 * reflections from the computational domain boundaries. Stacey boundaries
 * approximate an infinite medium by applying a local impedance condition.
 *
 * The Stacey condition relates the traction on the boundary to the velocity:
 * \f$
 *   \mathbf{t} = -\rho c_p \mathbf{v} \cdot \mathbf{n} \mathbf{n} - \rho c_s
 * (\mathbf{v} - (\mathbf{v} \cdot \mathbf{n})\mathbf{n})
 * \f$
 * where \f$\mathbf{t}\f$ is the traction, \f$\rho\f$ is density, \f$c_p\f$ and
 * \f$c_s\f$ are P-wave and S-wave velocities, \f$\mathbf{v}\f$ is particle
 * velocity, and \f$\mathbf{n}\f$ is the outward unit normal.
 *
 * This boundary type extends the acoustic free surface boundary by adding
 * geometric information necessary for Stacey condition evaluation: edge
 * integration weights and outward normal vectors to the domain boundary.
 *
 * @tparam DimensionTag Spatial dimension of the problem. For 2D problems, use
 *                      `specfem::dimension::type::dim2`.
 * @tparam UseSIMD Boolean flag for SIMD optimization. When enabled, allows
 *                 vectorized operations on multiple boundary points.
 *
 * @note Stacey boundaries are particularly effective for modeling semi-infinite
 *       domains in geophysical applications, such as regional earthquake
 * simulations.
 *
 * @see Clayton, R., & Engquist, B. (1977). Absorbing boundary conditions for
 * acoustic and elastic wave equations. Bulletin of the Seismological Society of
 * America.
 * @see specfem::boundary_conditions::stacey_type
 *
 * @code
 * // Example: Setting up a Stacey boundary
 * using StaceyBoundary = specfem::point::boundary<
 *     specfem::element::boundary_tag::stacey,
 *     specfem::dimension::type::dim2, false>;
 *
 * StaceyBoundary boundary;
 * boundary.tag += specfem::element::boundary_tag::stacey;
 * boundary.edge_weight = integration_weight;  // From quadrature rule
 * boundary.edge_normal(0) = normal_x;         // Outward normal components
 * boundary.edge_normal(1) = normal_y;
 * @endcode
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::stacey, DimensionTag, UseSIMD>
    : public boundary<specfem::element::boundary_tag::acoustic_free_surface,
                      DimensionTag, UseSIMD> {
private:
  constexpr static int num_dimensions =
      specfem::dimension::dimension<DimensionTag>::dim;
  /**
   * @name Private Typedefs
   *
   */
  ///@{
  using NormalViewType =
      specfem::datatype::VectorPointViewType<type_real, num_dimensions,
                                             UseSIMD>; ///< View type to store
                                                       ///< the normal vector to
                                                       ///< the edge at the
                                                       ///< quadrature point

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype; ///< SIMD
                                                                      ///< data
                                                                      ///< type
  ///@}

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  ///@}

  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto boundary_tag = specfem::element::boundary_tag::stacey;
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor for Stacey absorbing boundary.
   *
   * Creates a Stacey boundary object with default initialization:
   * - Edge weight set to zero
   * - Edge normal vector components initialized to zero
   * - Boundary tag uninitialized (should be set explicitly)
   *
   * @note Device/host portable constructor for Kokkos execution spaces.
   */
  KOKKOS_FUNCTION
  boundary() = default;

  /**
   * @brief Implicit conversion constructor from composite Stacey-Dirichlet
   * boundary.
   *
   * Extracts the Stacey boundary component from a composite boundary that
   * combines both Stacey absorbing and Dirichlet free surface conditions. This
   * constructor copies all relevant Stacey-specific data including boundary
   * tags, edge weights, and normal vectors.
   *
   * This conversion is critical in the SPECFEMPP boundary condition
   * decomposition process, where composite boundaries are split into their
   * constituent parts for specialized physical treatment.
   *
   * @param boundary Reference to a composite boundary containing both Stacey
   *                 and Dirichlet boundary condition data.
   *
   * @note Unlike the acoustic free surface conversion, this preserves all
   *       geometric information (edge weights and normals) required for
   *       Stacey boundary condition evaluation.
   */
  KOKKOS_FUNCTION
  boundary(const specfem::point::boundary<
           specfem::element::boundary_tag::composite_stacey_dirichlet,
           DimensionTag, UseSIMD> &boundary);
  ///@}

  datatype edge_weight = static_cast<type_real>(
      0.0); ///< Integration weight from the boundary surface
            ///< quadrature rule, used in Stacey condition
            ///< evaluation. Represents the differential area
            ///< element \f$dS\f$ for surface integrals.

  NormalViewType edge_normal = {
    static_cast<type_real>(0.0), static_cast<type_real>(0.0)
  }; ///< Outward unit normal vector \f$\mathbf{n}\f$ to the boundary surface
     ///< at the quadrature point. Essential for computing the impedance
     ///< matrix in Stacey boundary conditions. Components are ordered
     ///< as [n_x, n_y] for 2D problems or [n_x, n_y, n_z] for 3D.
};

/**
 * @brief Template specialization for composite Stacey-Dirichlet boundary
 * condition.
 *
 * This specialization represents the most complex boundary type in SPECFEMPP,
 * combining both Stacey absorbing boundary conditions and Dirichlet (acoustic
 * free surface) conditions in a single unified framework. Composite boundaries
 * are essential for handling complex geometrical configurations where different
 * physical processes occur simultaneously at boundary interfaces.
 *
 * The composite boundary condition seamlessly transitions between:
 * 1. **Stacey absorbing behavior**: Prevents artificial reflections by
 * simulating an infinite medium continuation
 * 2. **Dirichlet free surface behavior**: Enforces zero pressure/displacement
 *    conditions at free surfaces
 *
 * This design allows for sophisticated boundary treatment in scenarios such as:
 * - Fluid-solid interfaces with partial contact
 * - Topographic surfaces with varying boundary physics
 * - Multi-physics coupling regions
 *
 * The composite type inherits all functionality from the Stacey boundary
 * (including edge weights and normal vectors) while maintaining compatibility
 * with both constituent boundary types through implicit conversion operators.
 *
 * @tparam DimensionTag Spatial dimension specification for the boundary
 * geometry.
 * @tparam UseSIMD SIMD vectorization flag for performance optimization across
 *                 multiple boundary points.
 *
 * @note This is the highest level in the boundary inheritance hierarchy,
 *       encompassing all other boundary condition types.
 *
 * @see specfem::boundary_conditions::composite_stacey_dirichlet_type
 * @see boundary<specfem::element::boundary_tag::stacey, DimensionTag, UseSIMD>
 * @see boundary<specfem::element::boundary_tag::acoustic_free_surface,
 * DimensionTag, UseSIMD>
 *
 * @code
 * // Example: Composite boundary setup
 * using CompositeBoundary = specfem::point::boundary<
 *     specfem::element::boundary_tag::composite_stacey_dirichlet,
 *     specfem::dimension::type::dim2, false>;
 *
 * CompositeBoundary boundary;
 * // Setup includes both Stacey and free surface data
 * boundary.tag += specfem::element::boundary_tag::stacey;
 * boundary.tag += specfem::element::boundary_tag::acoustic_free_surface;
 * boundary.edge_weight = weight;
 * boundary.edge_normal = normal_vector;
 *
 * // Can be implicitly converted to constituent types:
 * auto stacey_part = static_cast<StaceyBoundary>(boundary);
 * auto free_surface_part = static_cast<FreeSurfaceBoundary>(boundary);
 * @endcode
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct boundary<specfem::element::boundary_tag::composite_stacey_dirichlet,
                DimensionTag, UseSIMD>
    : public boundary<specfem::element::boundary_tag::stacey, DimensionTag,
                      UseSIMD> {
public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  ///@}
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::composite_stacey_dirichlet;
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor for composite Stacey-Dirichlet boundary.
   *
   * Initializes a composite boundary object that can represent both Stacey
   * absorbing and acoustic free surface boundary conditions. All inherited
   * data members (boundary tags, edge weights, normal vectors) are initialized
   * to their default values.
   *
   * The composite boundary serves as the most general boundary type and can be
   * subsequently configured to represent any combination of supported boundary
   * physics through appropriate tag assignments.
   *
   * @note Portable across Kokkos execution spaces for heterogeneous computing.
   */
  KOKKOS_FUNCTION
  boundary() = default;
  ///@}
};

/**
 * @brief Conversion constructor implementation: Composite to Acoustic Free
 * Surface.
 *
 * Extracts the acoustic free surface component from a composite boundary by
 * copying only the boundary tag information. Edge-specific data (weights and
 * normals) used for Stacey conditions are intentionally omitted as they are not
 * relevant for free surface boundary processing.
 *
 * @tparam DimensionTag Spatial dimension template parameter.
 * @tparam UseSIMD SIMD vectorization template parameter.
 * @param boundary Source composite boundary containing mixed boundary condition
 * data.
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
KOKKOS_FUNCTION
specfem::point::boundary<specfem::element::boundary_tag::acoustic_free_surface,
                         DimensionTag, UseSIMD>::
    boundary(const specfem::point::boundary<
             specfem::element::boundary_tag::composite_stacey_dirichlet,
             DimensionTag, UseSIMD> &boundary) {
  this->tag = boundary.tag;
}

/**
 * @brief Conversion constructor implementation: Composite to Stacey.
 *
 * Extracts the Stacey absorbing boundary component from a composite boundary by
 * copying all relevant data including boundary tags, edge integration weights,
 * and outward normal vectors. This preserves complete information needed for
 * Stacey boundary condition evaluation.
 *
 * @tparam DimensionTag Spatial dimension template parameter.
 * @tparam UseSIMD SIMD vectorization template parameter.
 * @param boundary Source composite boundary containing complete boundary data.
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
KOKKOS_FUNCTION specfem::point::boundary<specfem::element::boundary_tag::stacey,
                                         DimensionTag, UseSIMD>::
    boundary(const specfem::point::boundary<
             specfem::element::boundary_tag::composite_stacey_dirichlet,
             DimensionTag, UseSIMD> &boundary) {
  this->tag = boundary.tag;
  this->edge_weight = boundary.edge_weight;
  this->edge_normal = boundary.edge_normal;
}

} // namespace point
} // namespace specfem
