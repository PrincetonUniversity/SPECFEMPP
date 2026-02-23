#pragma once

namespace specfem::data_access {

/**
 * @brief Data type classification for spectral element simulation components.
 *
 * Categorizes different data types used throughout SPECFEM simulations to
 * enable type-safe data access patterns and proper memory management
 * strategies.
 *
 * Used by data containers and access policies to determine appropriate handling
 * for mesh indices, physical properties, field variables, and computational
 * kernels.
 *
 * Each enum value is also associated with a type trait checker to facilitate
 * compile-time type detection and static assertions. Traits are typically
 * evaluated on container/accessor types that expose a `data_class` member.
 * For example:
 * @code{cpp}
 * struct PropertiesAccessor {
 *   static constexpr specfem::data_access::DataClassType data_class =
 *       specfem::data_access::DataClassType::properties;
 * };
 *
 * static_assert(
 *   specfem::data_access::is_properties<PropertiesAccessor>::value,
 *   "Accessor is not marked as properties"
 * );
 * @endcode
 */
enum DataClassType {
  index,                     ///< Element and point indices
  edge_index,                ///< Edge connectivity indices
  face_index,                ///< Face connectivity indices (3D)
  assembly_index,            ///< Global assembly indices
  mapped_index,              ///< Mapped/transformed indices
  properties,                ///< Material properties (density, moduli)
  kernels,                   ///< Computational kernels and sensitivities
  jacobian_matrix,           ///< Geometric transformation matrices
  field_derivatives,         ///< Spatial derivatives of fields
  displacement,              ///< Displacement field components
  velocity,                  ///< Velocity field components
  acceleration,              ///< Acceleration field components
  mass_matrix,               ///< Mass matrix coefficients
  source,                    ///< Source terms and excitations
  stress,                    ///< Stress tensor components
  stress_integrand,          ///< Stress integration quantities
  boundary,                  ///< Boundary condition data
  lagrange_derivative,       ///< Lagrange derivative operators
  weights,                   ///< Quadrature weights
  transfer_function_self,    ///< Self-coupling transfer functions
  transfer_function_coupled, ///< Cross-coupling transfer functions
  intersection_factor,       ///< Interface intersection factors
  intersection_normal,       ///< Interface normal vectors
  nonconforming_interface,   ///< Non-conforming mesh interfaces
  conforming_interface,      ///< Conforming mesh interfaces
  global_coordinates,        ///< Global coordinate data
  memory,                    ///< Memory variables for attenuation computation
  attenuation_factor         ///< Attenuation factors for memory variable update
};
} // namespace specfem::data_access
