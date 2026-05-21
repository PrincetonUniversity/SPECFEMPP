#pragma once

#include "specfem/data_access.hpp"
#include "specfem/datatype.hpp"
#include "specfem/enums.hpp"
#include "specfem/point/jacobian_matrix.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {
/**
 * @class stress_integrand
 * @brief Stress tensor integrated with Jacobian in reference element
 * coordinates
 *
 * Represents the stress tensor integrated with respect to the reference element
 * Jacobian determinant. This class manages stress components in the reference
 * (ξ, η, γ) coordinate system and provides transformation to physical (x, y, z)
 * coordinates.
 *
 * **Stress Transformation Mathematics:**
 *
 * The stress tensor transformation relates stress in physical coordinates T(σ)
 * to stress in reference coordinates F through the Jacobian matrix J.
 *
 * **2D Transformation (ξ, γ) ↔ (x, z):**
 *
 * Forward (physical → reference):
 * \f$ F(i,0) = |J| \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial x} +
 *                           T(i,1) \cdot \frac{\partial\gamma}{\partial x}) \f$
 * \f$ F(i,1) = |J| \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial z} +
 *                           T(i,1) \cdot \frac{\partial\gamma}{\partial z}) \f$
 *
 * Inverse (reference → physical):
 * \f$ T(i,0) = |J^{-1}| \cdot (F(i,0) \cdot \frac{\partial x}{\partial\xi} +
 *                                 F(i,1) \cdot \frac{\partial
 * x}{\partial\gamma}) \f$
 * \f$ T(i,1) = |J^{-1}| \cdot (F(i,0) \cdot \frac{\partial z}{\partial\xi} +
 *                                 F(i,1) \cdot \frac{\partial
 * z}{\partial\gamma}) \f$
 *
 * **3D Transformation (ξ, η, γ) ↔ (x, y, z):**
 *
 * Forward transformation:
 * \f$ F(i,0) = |J| \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial x} +
 *                           T(i,1) \cdot \frac{\partial\eta}{\partial x} +
 *                           T(i,2) \cdot \frac{\partial\gamma}{\partial x}) \f$
 * \f$ F(i,1) = |J| \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial y} +
 *                           T(i,1) \cdot \frac{\partial\eta}{\partial y} +
 *                           T(i,2) \cdot \frac{\partial\gamma}{\partial y}) \f$
 * \f$ F(i,2) = |J| \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial z} +
 *                           T(i,1) \cdot \frac{\partial\eta}{\partial z} +
 *                           T(i,2) \cdot \frac{\partial\gamma}{\partial z}) \f$
 *
 * Inverse transformation:
 * \f$ T(i,0) = |J^{-1}| \cdot (F(i,0) \cdot \frac{\partial x}{\partial\xi} +
 *                                 F(i,1) \cdot \frac{\partial x}{\partial\eta}
 * + F(i,2) \cdot \frac{\partial x}{\partial\gamma}) \f$
 * \f$ T(i,1) = |J^{-1}| \cdot (F(i,0) \cdot \frac{\partial y}{\partial\xi} +
 *                                 F(i,1) \cdot \frac{\partial y}{\partial\eta}
 * + F(i,2) \cdot \frac{\partial y}{\partial\gamma}) \f$
 * \f$ T(i,2) = |J^{-1}| \cdot (F(i,0) \cdot \frac{\partial z}{\partial\xi} +
 *                                 F(i,1) \cdot \frac{\partial z}{\partial\eta}
 * + F(i,2) \cdot \frac{\partial z}{\partial\gamma}) \f$
 *
 * where |J| is the Jacobian determinant and |J^{-1}| is its inverse.
 *
 * **Implementation Note:**
 * Coordinate transformations are now handled through generic matrix operations
 * (`operator*` in TensorPointViewType) and the `specfem::algorithms::inverse()`
 * function for matrix inversion. This provides a cleaner separation of concerns
 * between stress representation and coordinate transformation.
 *
 * @tparam T Scalar floating-point type
 * @tparam Dimension Spatial dimension (2 or 3)
 * @tparam UseSIMD Boolean flag for SIMD vectorization
 */

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress_integrand
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::stress_integrand, DimensionTag,
          UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::stress_integrand, DimensionTag,
      UseSIMD>; ///< Base accessor
                ///< type
public:
  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int dimension =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD type
  using value_type =
      typename base_type::template tensor_type<type_real, components,
                                               dimension>; ///< Underlying view
                                                           ///< type to store
                                                           ///< the stress
                                                           ///< integrand
  ///@}

  value_type F; ///< View to store the stress integrand

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION stress_integrand() = default;

  /**
   * @brief Constructor
   *
   * @param F Stress integrands
   */
  KOKKOS_FUNCTION stress_integrand(const value_type &F) : F(F) {}
  ///@}
};

} // namespace point
} // namespace specfem
