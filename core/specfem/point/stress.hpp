#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/interface.hpp"
#include "jacobian_matrix.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Represents a stress tensor at a quadrature point in spectral element
 * simulations.
 *
 * The stress class encapsulates stress tensor data and provides operations for
 * stress transformations in finite element computations. The tensor dimensions
 * and components are determined by the medium type and spatial dimension:
 * - Acoustic medium: 1 component in 2D/3D (scalar pressure)
 * - Elastic medium: 2 components in 2D, 3 components in 3D (velocity
 * components)
 * - Poroelastic medium: 4 components in 2D (solid and fluid phases)
 *
 * The class inherits from the data access framework providing SIMD
 * vectorization support and efficient memory access patterns for
 * high-performance computing.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam MediumTag Physical medium type (acoustic, elastic_psv, elastic,
 * poroelastic)
 * @tparam UseSIMD Enable SIMD vectorization for performance optimization
 *
 * @code
 * // Example: Create stress tensor for 2D elastic medium
 * using stress_type = specfem::point::stress<specfem::dimension::type::dim2,
 *                                           specfem::element::medium_tag::elastic_psv,
 *                                           false>;
 *
 * // Initialize stress components (2x2 tensor for 2D elastic)
 * typename stress_type::value_type T(1.1, 2.1,  // first column  (component 0,
 * 1) 1.2, 2.2); // second column (component 0, 1) stress_type stress_tensor(T);
 *
 * // Transform stress using jacobian matrix
 * // auto jacobian = initialize_jacobian_matrix();
 * auto transformed_stress = stress_tensor * jacobian;
 * @endcode
 *
 * @see specfem::point::jacobian_matrix
 * @see specfem::data_access::Accessor
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::stress, DimensionTag, UseSIMD> {
private:
  /** @brief Base accessor type for data access framework integration */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::stress, DimensionTag, UseSIMD>;

public:
  /**
   * @name Static Properties
   * @brief Compile-time constants derived from template parameters
   */
  ///@{
  /** @brief Spatial dimension (2 or 3) */
  constexpr static int dimension =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;

  /** @brief Number of stress components based on medium type */
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  /** @brief Template parameter for spatial dimension */
  constexpr static specfem::dimension::type dimension_tag = DimensionTag;

  /** @brief Template parameter for medium type */
  constexpr static specfem::element::medium_tag medium_tag = MediumTag;

  /** @brief Template parameter for SIMD usage */
  constexpr static bool using_simd = UseSIMD;
  ///@}

  /**
   * @name Type Definitions
   * @brief Type aliases for SIMD and tensor operations
   */
  ///@{
  /** @brief SIMD type for vectorized operations */
  using simd = typename base_type::template simd<type_real>;

  /** @brief Tensor type for storing stress components (components × dimension)
   */
  using value_type =
      typename base_type::template tensor_type<type_real, components,
                                               dimension>;
  ///@}

  /**
   * @name Data Members
   */
  ///@{
  /** @brief Stress tensor storage with shape (components × dimension) */
  value_type T;
  ///@}

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Default constructor.
   *
   * Initializes stress tensor with default values (typically zero).
   */
  KOKKOS_FUNCTION stress() = default;

  /**
   * @brief Constructor with stress tensor initialization.
   *
   * @param T Stress tensor with components arranged as (components × dimension)
   *
   * @code
   * // For 2D elastic medium (2 components × 2 dimensions)
   * typename stress_type::value_type tensor(1.1, 2.1,  // component 0: (σxx,
   * σxz) 1.2, 2.2); // component 1: (σzx, σzz) stress_type stress(tensor);
   * @endcode
   */
  KOKKOS_FUNCTION stress(const value_type &T) : T(T) {}
  ///@}

  /**
   * @name Operators
   */
  ///@{
  /**
   * @brief Transform stress tensor using jacobian matrix.
   *
   * Applies the coordinate transformation from reference element to physical
   * element using the jacobian matrix. This operation transforms stress
   * components from the reference (ξ, ζ) coordinate system to the physical (x,
   * z) coordinate system.
   *
   * The transformation formula for 2D is:
   * \f$ F(i,0) = J \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial x} + T(i,1)
   * \cdot \frac{\partial\zeta}{\partial x}) \f$
   * \f$ F(i,1) = J \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial z} + T(i,1)
   * \cdot \frac{\partial\zeta}{\partial z}) \f$
   *
   * where \f$ J \f$ is the jacobian determinant and the partial derivatives are
   * the inverse jacobian matrix elements.
   *
   * @param jacobian_matrix Jacobian matrix containing transformation
   * derivatives
   * @return Transformed stress tensor in physical coordinates
   *
   * @code
   * stress_type stress(stress_tensor);
   * auto jacobian = compute_jacobian_matrix(quadrature_point);
   * auto transformed = stress * jacobian;
   * @endcode
   */
  KOKKOS_INLINE_FUNCTION
  value_type operator*(
      const specfem::point::jacobian_matrix<specfem::dimension::type::dim2,
                                            true, UseSIMD> &jacobian_matrix)
      const {
    value_type F;

    for (int icomponent = 0; icomponent < components; ++icomponent) {
      F(icomponent, 0) =
          jacobian_matrix.jacobian * (T(icomponent, 0) * jacobian_matrix.xix +
                                      T(icomponent, 1) * jacobian_matrix.xiz);
      F(icomponent, 1) = jacobian_matrix.jacobian *
                         (T(icomponent, 0) * jacobian_matrix.gammax +
                          T(icomponent, 1) * jacobian_matrix.gammaz);
    }

    return F;
  }

  /**
   * @brief Transform stress tensor using 3D jacobian matrix.
   *
   * Applies the coordinate transformation from reference element to physical
   * element using the jacobian matrix. This operation transforms stress
   * components from the reference (ξ, η, ζ) coordinate system to the physical
   * (x, y, z) coordinate system.
   *
   * The transformation formula for 3D is:
   * \f$ F(i,0) = J \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial x} +
   *                       T(i,1) \cdot \frac{\partial\eta}{\partial x} +
   *                       T(i,2) \cdot \frac{\partial\zeta}{\partial x}) \f$
   * \f$ F(i,1) = J \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial y} +
   *                       T(i,1) \cdot \frac{\partial\eta}{\partial y} +
   *                       T(i,2) \cdot \frac{\partial\zeta}{\partial y}) \f$
   * \f$ F(i,2) = J \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial z} +
   *                       T(i,1) \cdot \frac{\partial\eta}{\partial z} +
   *                       T(i,2) \cdot \frac{\partial\zeta}{\partial z}) \f$
   *
   * where \f$ J \f$ is the jacobian determinant and the partial derivatives are
   * the inverse jacobian matrix elements.
   *
   * @param jacobian_matrix 3D Jacobian matrix containing transformation
   * derivatives
   * @return Transformed stress tensor in physical coordinates
   *
   * @code
   * stress_type stress(stress_tensor);
   * auto jacobian = compute_jacobian_matrix_3d(quadrature_point);
   * auto transformed = stress * jacobian;
   * @endcode
   */
  KOKKOS_INLINE_FUNCTION
  value_type operator*(
      const specfem::point::jacobian_matrix<specfem::dimension::type::dim3,
                                            true, UseSIMD> &jacobian_matrix)
      const {
    value_type F;

    for (int icomponent = 0; icomponent < components; ++icomponent) {
      F(icomponent, 0) =
          jacobian_matrix.jacobian * (T(icomponent, 0) * jacobian_matrix.xix +
                                      T(icomponent, 1) * jacobian_matrix.xiy +
                                      T(icomponent, 2) * jacobian_matrix.xiz);
      F(icomponent, 1) =
          jacobian_matrix.jacobian * (T(icomponent, 0) * jacobian_matrix.etax +
                                      T(icomponent, 1) * jacobian_matrix.etay +
                                      T(icomponent, 2) * jacobian_matrix.etaz);
      F(icomponent, 2) = jacobian_matrix.jacobian *
                         (T(icomponent, 0) * jacobian_matrix.gammax +
                          T(icomponent, 1) * jacobian_matrix.gammay +
                          T(icomponent, 2) * jacobian_matrix.gammaz);
    }

    return F;
  }

  /**
   * @brief Equality comparison operator.
   *
   * Compares two stress tensors for equality by comparing their underlying
   * tensor data element-wise.
   *
   * @param other Another stress tensor to compare with
   * @return true if tensors are equal, false otherwise
   */
  KOKKOS_INLINE_FUNCTION
  bool operator==(const stress &other) const { return T == other.T; };
  ///@}

  /**
   * @name Utility Functions
   */
  ///@{
  /**
   * @brief Generate string representation of the stress tensor.
   *
   * Creates a formatted string showing all components of the stress tensor
   * for debugging and visualization purposes. The output format shows each
   * component with its (component, dimension) indices.
   *
   * @return Formatted string representation of the stress tensor
   *
   * @code
   * stress_type stress(tensor_data);
   * std::cout << stress.print() << std::endl;
   * // Output:
   * // Stress Tensor:
   * // T(0, 0) = 1.1, T(0, 1) = 1.2
   * // T(1, 0) = 2.1, T(1, 1) = 2.2
   * @endcode
   */
  std::string print() const {
    std::ostringstream oss;
    oss << "Stress Tensor:\n";
    for (int i = 0; i < components; ++i) {
      oss << "T(" << i << ", 0) = " << T(i, 0) << ", "
          << "T(" << i << ", 1) = " << T(i, 1) << "\n";
    }
    return oss.str();
  }
  ///@}
};
} // namespace point
} // namespace specfem
