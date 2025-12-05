#pragma once

#include "impl/nonconforming_transfer_function.hpp"

namespace specfem::point {

/**
 * @brief Transfer function template for non-conforming interface coupling at
 quadrature points.
 *
 * This primary template defines the interface for transfer functions that
 handle field
 * interpolation and data exchange between non-conforming mesh interfaces in
 spectral
 * element simulations. Transfer functions are fundamental to mortar-based
 coupling
 * methods that enable flexible mesh generation while preserving solution
 accuracy.

 * **Mortar Method Implementation:**
 * The transfer function encapsulates mortar integral computations:
 * \f$
 *   \int_{\Gamma} N_i^{\text{mortar}} N_j^{\text{slave}} d\Gamma \quad
 \text{and} \quad
 *   \int_{\Gamma} N_i^{\text{mortar}} N_j^{\text{master}} d\Gamma
 * \f$
 * where \f$N_i^{\text{mortar}}\f$ are mortar basis functions and \f$N_j\f$ are
 element basis functions.
 *
 * @tparam IsSelf Boolean indicating whether this represents the self side or
 coupled side
 *                of the interface. Affects the direction of data transfer and
 the
 *                interpretation of interface coupling conditions.
 *
 * @tparam NQuadIntersection Number of quadrature points used for mortar
 integral evaluation.
 *                          Higher values provide more accurate interface
 integration but
 *                          increase computational cost.
 *
 * @tparam DimensionTag Spatial dimension of the interface geometry:
 *                      - `specfem::dimension::type::dim2`: 2D interfaces (edge
 coupling)
 *                      - `specfem::dimension::type::dim3`: 3D interfaces
 (surface coupling)
 *
 * @tparam ConnectionTag Type of interface connection defining coupling
 behavior:
 *                       - `specfem::connections::type::conforming`: Matching
 mesh interfaces
 *                       - `specfem::connections::type::nonconforming`:
 Mismatched mesh interfaces
 *
 * @tparam InterfaceTag Physical interface type determining coupling physics:
 *                      - `specfem::interface::acoustic_elastic`: Fluid-solid
 coupling
 *                      - `specfem::interface::elastic_acoustic`: Solid-fluid
 coupling
 *                      - `specfem::interface::elastic_elastic`: Solid-solid
 coupling
 *
 * @tparam BoundaryTag Boundary condition type applied at the interface:
 *                     - `specfem::element::boundary_tag::none`: No special
 boundary condition
 *                     - `specfem::element::boundary_tag::stacey`: Absorbing
 boundary
 *                     -
 `specfem::element::boundary_tag::acoustic_free_surface`: Free surface
 *
 * @note This is a primary template declaration. Actual implementation is
 provided through
 *       template specializations for specific combinations of parameters.
 *
 * @see specfem::mortar::projection_operator
 * @see specfem::interface::coupling_matrix
 * @see impl::nonconforming_transfer_function
 *
 * @code
 * // Example: 2D acoustic-elastic interface with mortar coupling
 * using TransferFunc = specfem::point::nonconforming_transfer_function<
 *     true,  // Self side of interface
 *     4,     // 4 quadrature points for mortar integration
 *     specfem::dimension::type::dim2,
 *     specfem::connections::type::nonconforming,
 *     specfem::interface::acoustic_elastic,
 *     specfem::element::boundary_tag::none>;
 *
 * TransferFunc transfer_function(mortar_weights, projection_matrix);
 *
 * // Use for field transfer across interface
 * auto target_field = transfer_function.apply_projection(source_field);
 * @endcode
 */
template <bool IsSelf, int NQuadIntersection,
          specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct nonconforming_transfer_function;

/**
 * @brief 2D non-conforming transfer function specialization for interface
 * coupling.
 *
 * This template specialization provides a concrete implementation of transfer
 * functions for 2D non-conforming interface coupling in spectral element
 * simulations. It handles the complex mathematical operations required for
 * accurate field transfer across mismatched mesh interfaces using mortar-based
 * projection methods.
 *
 * **Implementation Details:**
 * This specialization inherits from both the data access framework and the
 * implementation class, providing a complete interface for:
 * - **Mortar integration**: Computing projection operators via numerical
 * quadrature
 * - **Field interpolation**: Transferring field values between non-conforming
 * meshes
 * - **Interface coupling**: Enforcing weak continuity conditions across
 * material boundaries
 * - **Load distribution**: Properly distributing interface forces and fluxes
 *
 * **Mortar Method Theory:**
 * For non-conforming interfaces, the mortar method introduces a Lagrange
 * multiplier field
 * \f$\lambda\f$ defined on the mortar surface to enforce interface conditions:
 * \f$
 *   \int_{\Gamma_m} \lambda \cdot [[\mathbf{u}]] d\Gamma_m = 0
 * \f$
 * where \f$[[\mathbf{u}]]\f$ represents the jump in the field across the
 * interface.
 *
 * **Projection Operators:**
 * The transfer function implements discrete projection operators:
 * \f$
 *   \mathbf{D} = \int_{\Gamma_m} \mathbf{N}_m \mathbf{N}_s^T d\Gamma_m \quad
 * \text{(slave side)}
 * \f$
 * \f$
 *   \mathbf{M} = \int_{\Gamma_m} \mathbf{N}_m \mathbf{N}_m^T d\Gamma_m \quad
 * \text{(mortar mass matrix)}
 * \f$
 * where \f$\mathbf{N}_m\f$ are mortar basis functions and \f$\mathbf{N}_s\f$
 * are slave element basis functions.
 *
 * **Template Specialization Parameters:**
 * This specialization is specifically designed for:
 * - **2D geometry**: Interfaces are curves (edges) requiring 1D mortar
 * integration
 * - **Non-conforming coupling**: Mesh discretizations do not align at
 * interfaces
 * - **Point-wise operations**: Operations performed at individual quadrature
 * points
 * - **Multi-physics support**: Handles various interface types
 * (acoustic-elastic, etc.)
 *
 * @tparam IsSelf Boolean flag indicating interface side:
 *                - `true`: Self side (typically the "master" side in mortar
 * methods)
 *                - `false`: Coupled side (typically the "slave" side)
 *
 * @tparam NQuadIntersection Number of quadrature points for mortar surface
 * integration. Must be sufficient to accurately integrate polynomial products
 *                          of order up to N_mortar + N_element.
 *
 * @tparam InterfaceTag Physical coupling type determining field exchange rules:
 *                      - Acoustic-elastic: Pressure-displacement coupling
 *                      - Elastic-acoustic: Displacement-pressure coupling
 *                      - Elastic-elastic: Displacement-displacement coupling
 *
 * @tparam BoundaryTag Additional boundary condition applied at the interface,
 *                     such as absorbing conditions or free surface constraints.
 *
 * @note The specialization provides compile-time constants for interface
 * identification and inherits all necessary functionality from the
 * implementation base class.
 *
 * @see specfem::mortar::basis_functions
 * @see specfem::quadrature::mortar_integration
 * @see specfem::interface::coupling_conditions
 *
 * @code
 * // Example: Acoustic-elastic coupling with 4-point quadrature
 * using AcousticElasticTransfer =
 * specfem::point::nonconforming_transfer_function< true,  // Self (acoustic)
 * side 4,     // 4-point Gauss quadrature on mortar
 *     specfem::dimension::type::dim2,
 *     specfem::connections::type::nonconforming,
 *     specfem::interface::acoustic_elastic,
 *     specfem::element::boundary_tag::none>;
 *
 * // Initialize transfer function with mortar data
 * AcousticElasticTransfer transfer_func(mortar_weights, coupling_matrix);
 *
 * // Interface coupling workflow:
 * // 1. Extract fields from both sides
 * auto acoustic_pressure = acoustic_element.get_pressure_field();
 * auto elastic_displacement = elastic_element.get_displacement_field();
 *
 * // 2. Apply transfer functions for coupling
 * auto projected_pressure = transfer_func.project_to_mortar(acoustic_pressure);
 * auto interface_traction =
 * transfer_func.compute_interface_traction(projected_pressure);
 *
 * // 3. Apply interface conditions
 * elastic_element.apply_interface_traction(interface_traction);
 * acoustic_element.apply_interface_velocity(elastic_displacement);
 * @endcode
 */
template <bool IsSelf, int NQuadIntersection,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct nonconforming_transfer_function<
    IsSelf, NQuadIntersection, specfem::dimension::type::dim2,
    specfem::connections::type::nonconforming, InterfaceTag, BoundaryTag>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2, false>,
      public impl::nonconforming_transfer_function<
          IsSelf, NQuadIntersection, specfem::dimension::type::dim2> {
private:
  /**
   * @brief Base accessor type alias for data access framework integration.
   *
   * Provides the foundation for type-safe data access and SPECFEMPP framework
   * integration, enabling seamless data exchange with the assembly system.
   */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_edge,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2, false>;

  /**
   * @brief Implementation type alias for mortar method functionality.
   *
   * Encapsulates the mathematical algorithms and data structures required
   * for non-conforming transfer operations, including projection operators
   * and interface coupling computations.
   */
  using impl_type =
      impl::nonconforming_transfer_function<IsSelf, NQuadIntersection,
                                            specfem::dimension::type::dim2>;

public:
  /**
   * @name Compile-time Interface Identification
   * @{
   */

  /**
   * @brief Interface physics type for coupling condition determination.
   *
   * Specifies the physical coupling behavior (acoustic-elastic,
   * elastic-elastic, etc.) which determines the mathematical form of interface
   * conditions and field exchange rules.
   */
  static constexpr auto interface_tag = InterfaceTag;

  /**
   * @brief Connection type indicating non-conforming mesh interface.
   *
   * Identifies this as a non-conforming interface requiring mortar-based
   * coupling methods rather than direct point-to-point field transfer.
   */
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;

  /**
   * @brief Boundary condition type applied at the interface.
   *
   * Specifies additional boundary physics (absorbing, free surface, etc.) that
   * may be superimposed on the interface coupling conditions.
   */
  static constexpr auto boundary_tag = BoundaryTag;

  /**
   * @brief Number of quadrature points for mortar surface integration.
   *
   * Determines the accuracy of mortar integral evaluation and must be chosen
   * to exactly integrate polynomial products arising in the mortar method.
   */
  static constexpr auto n_quad_intersection = NQuadIntersection;

  /** @} */

  /**
   * @brief Variadic constructor for flexible transfer function initialization.
   *
   * Forwards construction arguments to the implementation class, enabling
   * initialization with various combinations of mortar weights, projection
   * matrices, and interface geometry data.
   *
   * @param args Forwarded arguments for implementation constructor. Typical
   *             arguments include mortar quadrature weights, coupling matrices,
   *             and interface geometric data.
   *
   * @note This constructor is device/host portable for Kokkos execution spaces.
   *
   * @code
   * // Example construction with mortar data
   * auto weights = compute_mortar_weights();
   * auto coupling_matrix = build_coupling_matrix();
   * auto geometry = extract_interface_geometry();
   *
   * TransferFunction transfer_func(weights, coupling_matrix, geometry);
   * @endcode
   */
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION nonconforming_transfer_function(const Args &...args)
      : impl_type(args...) {}
};
} // namespace specfem::point
