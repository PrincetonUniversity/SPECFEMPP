#pragma once

#include "datatypes/simd.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::point::impl {

/**
 * @brief Type-safe field data container for spectral element quadrature points.
 *
 * Stores physical field values with compile-time type information about
 * dimension, medium type, and data class. Supports optional SIMD vectorization.
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 * @tparam MediumTag Physical medium (acoustic/elastic/poroelastic)
 * @tparam DataClass Field type (displacement/velocity/acceleration/mass_matrix)
 * @tparam UseSIMD Enable SIMD vectorization
 *
 * @code
 * // 2D elastic displacement field
 * using DispField = field<dim2, elastic, displacement, false>;
 * DispField u(0.0, 0.0);  // Initialize to zero
 * u(0) = 1.5;  // x-component
 * u(1) = 2.3;  // z-component
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass, bool UseSIMD>
class field : public specfem::data_access::Accessor<
                  specfem::data_access::AccessorType::point, DataClass,
                  DimensionTag, UseSIMD> {
private:
  /// @brief Type alias for the base accessor class
  using base_type =
      specfem::data_access::Accessor<specfem::data_access::AccessorType::point,
                                     DataClass, DimensionTag, UseSIMD>;

public:
  /// @brief Number of field components (acoustic: 1, elastic 2D: 2, elastic 3D:
  /// 3)
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  /// @brief SIMD type for vectorized operations
  using simd = typename base_type::template simd<type_real>;

  /// @brief Vector type storing field component values
  using value_type =
      typename base_type::template vector_type<type_real, components>;

  /// @brief Physical medium type identifier
  constexpr static auto medium_tag = MediumTag;

private:
  /**
   * @brief Internal storage for field component values.
   *
   * Stores the actual field data using the optimized value_type container.
   * The storage layout and access patterns are determined by the template
   * parameters and base class implementation.
   *
   * @note Direct access to m_data should be avoided; use the public
   *       accessor functions instead for type safety and optimization.
   */
  value_type m_data;

public:
  /**
   * @brief Default constructor - creates field with uninitialized values.
   */
  KOKKOS_FORCEINLINE_FUNCTION field() = default;

  /**
   * @brief Access internal field data storage.
   *
   * @return const reference to the internal value_type storing field components
   */
  KOKKOS_FORCEINLINE_FUNCTION const value_type &get_data() const {
    return m_data;
  }

  /**
   * @brief Construct field with uniform initialization across all components.
   *
   * Initializes all field components to the same scalar value. This is useful
   * for creating zero-initialized fields or fields with uniform values.
   *
   * @tparam U Type convertible to the field's component type
   * @param initializer The value to assign to all field components
   *
   * @pre U must be convertible to typename value_type::value_type
   *
   * @code{.cpp}
   * // Create zero-initialized displacement field
   * DisplacementField u_field(0.0);
   *
   * // Create field with uniform value
   * VelocityField v_field(1.5);  // All components = 1.5
   * @endcode
   */
  template <
      typename U,
      std::enable_if_t<
          std::is_convertible_v<U, typename value_type::value_type>, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION constexpr field(const U initializer) {
    for (std::size_t icomp = 0; icomp < components; ++icomp)
      m_data(icomp) = initializer;
  }

  /**
   * @brief Construct field from value_type object.
   *
   * Directly initializes the field from a compatible value_type object.
   * This allows construction from pre-computed vector objects.
   *
   * @tparam U Type that must match value_type exactly
   * @param initializer The value_type object to copy from
   *
   * @pre U must be exactly the same type as value_type
   */
  template <typename U, typename... Args,
            typename = std::enable_if_t<std::is_same_v<U, value_type>, int> >
  KOKKOS_FORCEINLINE_FUNCTION constexpr field(const U &initializer)
      : m_data(initializer) {}

  /**
   * @brief Construct field with component-wise initialization.
   *
   * Allows direct initialization of field components by providing individual
   * values for each component. The number of arguments must exactly match
   * the number of field components.
   *
   * @tparam Args Types of the component values (must match component count)
   * @param args Individual values for each field component
   *
   * @pre sizeof...(Args) must equal the number of field components
   *
   * @code{.cpp}
   * // 2D displacement field: ux, uz
   * DisplacementField u_field(1.5, 2.3);
   *
   * // 3D velocity field: vx, vy, vz
   * VelocityField v_field(0.1, 0.2, 0.3);
   * @endcode
   */
  template <typename... Args,
            typename = std::enable_if_t<sizeof...(Args) == components> >
  KOKKOS_FORCEINLINE_FUNCTION constexpr field(Args &&...args)
      : m_data(std::forward<Args>(args)...) {}

  /**
   * @brief Access field component by index (const version).
   *
   * Provides read-only access to individual field components using zero-based
   * indexing. For 2D problems: 0=x, 1=z. For 3D problems: 0=x, 1=y, 2=z.
   *
   * @param icomp Component index (0 to components-1)
   * @return const reference to the component value
   *
   * @pre icomp must be less than the number of components
   *
   * @code{.cpp}
   * DisplacementField u_field(1.5, 2.3);
   * auto ux = u_field(0);  // x-component = 1.5
   * auto uz = u_field(1);  // z-component = 2.3
   * @endcode
   */
  KOKKOS_FORCEINLINE_FUNCTION const typename value_type::value_type &
  operator()(const std::size_t icomp) const {
    return m_data(icomp);
  }

  /**
   * @brief Access field component by index (mutable version).
   *
   * Provides read-write access to individual field components using zero-based
   * indexing. Allows modification of field component values.
   *
   * @param icomp Component index (0 to components-1)
   * @return mutable reference to the component value
   *
   * @pre icomp must be less than the number of components
   *
   * @code{.cpp}
   * DisplacementField u_field;
   * u_field(0) = 1.5;  // Set x-component
   * u_field(1) = 2.3;  // Set z-component
   * @endcode
   */
  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type &
  operator()(const std::size_t icomp) {
    return m_data(icomp);
  }

  /**
   * @brief Equality comparison operator.
   *
   * Compares two field objects for exact equality by comparing their
   * internal data storage.
   *
   * @param other The field object to compare against
   * @return true if all components are exactly equal, false otherwise
   */
  KOKKOS_FORCEINLINE_FUNCTION bool operator==(const field &other) const {
    return (this->m_data == other.m_data);
  }

  /**
   * @brief Inequality comparison operator.
   *
   * Compares two field objects for inequality.
   *
   * @param other The field object to compare against
   * @return true if any component differs, false if all components are equal
   */
  KOKKOS_FORCEINLINE_FUNCTION bool operator!=(const field &other) const {
    return !(*this == other);
  }

  /**
   * @brief Multiplication assignment operator with constant value.
   *
   * Multiply by a constant value and assign the result to this field.
   *
   * @param other The factor to be multiplied with
   * @return reference to this field after multiplication
   */
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto &
  operator*=(const typename value_type::value_type &other) {
    this->m_data *= other;
    return *this;
  }

  /**
   * @brief Output a string representation of the field components.
   *
   * Outputs the values of all field components to the specified output stream.
   * Each component is printed in order, enclosed in square brackets.
   * If SIMD the SIMD values are printed in curly brackets.
   *
   */
  std::string print() const {
    std::ostringstream os;
    os << "{";
    for (std::size_t i = 0; i < components; ++i) {
      os << this->m_data(i);
      if (i < components - 1) {
        os << ",\n";
      }
    }
    os << "}";
    return os.str();
  }
};

} // namespace specfem::point::impl
