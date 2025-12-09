#pragma once

#include "datatypes/simd.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::point::impl {

/**
 * @brief Point field accessor for spectral element field data storage and
 * manipulation.
 *
 * This class provides a strongly-typed interface for accessing and manipulating
 * field data at individual points within spectral elements. It serves as a data
 * container that holds field values (displacement, velocity, acceleration,
 * etc.) with compile-time knowledge of the spatial dimension, medium type, data
 * class, and SIMD optimization settings.
 *
 * The field class is designed to work seamlessly with the SPECFEM++ assembly
 * system, providing type-safe access to field components while maintaining
 * optimal performance through Kokkos integration and optional SIMD
 * vectorization.
 *
 * @tparam DimensionTag The spatial dimension (dim2 or dim3) of the field
 * @tparam MediumTag The medium type (acoustic, elastic, poroelastic, etc.)
 * @tparam DataClass The type of field data (displacement, velocity,
 * acceleration, mass_matrix)
 * @tparam UseSIMD Whether to enable SIMD vectorization for performance
 * optimization
 *
 * @note This class inherits from specfem::data_access::Accessor to provide
 * consistent interface and type traits for the SPECFEM++ data access system.
 *
 * @code{.cpp}
 * // Example: Creating displacement field accessors for 2D elastic medium
 * using DisplacementField = specfem::point::impl::field<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::elastic,
 *     specfem::data_access::DataClassType::displacement,
 *     false>;  // No SIMD
 *
 * // Initialize with zero displacement
 * DisplacementField u_field(0.0);
 *
 * // Set displacement components
 * u_field(0) = 1.5;  // x-component
 * u_field(1) = 2.3;  // z-component (2D)
 *
 * // Access displacement values
 * auto ux = u_field(0);
 * auto uz = u_field(1);
 * @endcode
 *
 * @code{.cpp}
 * // Example: Creating velocity field with SIMD optimization
 * using VelocityField = specfem::point::impl::field<
 *     specfem::dimension::type::dim3,
 *     specfem::element::medium_tag::acoustic,
 *     specfem::data_access::DataClassType::velocity,
 *     true>;   // Enable SIMD
 *
 * // Initialize velocity field with specific components
 * VelocityField v_field(1.0, 2.0, 3.0);  // vx, vy, vz
 *
 * // Use in assembly operations
 * specfem::assembly::load_on_device(point_index, field_container, v_field);
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass, bool UseSIMD>
class field : public specfem::data_access::Accessor<
                  specfem::data_access::AccessorType::point, DataClass,
                  DimensionTag, UseSIMD> {
private:
  /**
   * @brief Base accessor type.
   */
  using base_type =
      specfem::data_access::Accessor<specfem::data_access::AccessorType::point,
                                     DataClass, DimensionTag, UseSIMD>;

public:
  /**
   * @brief Number of field components.
   */
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  /**
   * @brief SIMD type for vectorized operations.
   */
  using simd = typename base_type::template simd<type_real>;

  /**
   * @brief Vector type for storing field component values.
   */
  using value_type =
      typename base_type::template vector_type<type_real, components>;

  /**
   * @brief Medium tag identifying the physical medium type.
   */
  constexpr static auto medium_tag = MediumTag;

private:
  /**
   * @brief Internal storage for field component values.
   */
  value_type m_data;

public:
  /**
   * @brief Default constructor.
   */
  KOKKOS_FORCEINLINE_FUNCTION field() = default;

  /**
   * @brief Access internal field data storage.
   *
   * @return Const reference to the internal data.
   */
  KOKKOS_FORCEINLINE_FUNCTION const value_type &get_data() const {
    return m_data;
  }

  /**
   * @brief Construct field with uniform initialization.
   *
   * @tparam U Type convertible to the component type.
   * @param initializer The value to assign to all components.
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
   * @tparam U Type matching value_type.
   * @param initializer The value object to copy from.
   */
  template <typename U, typename... Args,
            typename = std::enable_if_t<std::is_same_v<U, value_type>, int> >
  KOKKOS_FORCEINLINE_FUNCTION constexpr field(const U &initializer)
      : m_data(initializer) {}

  /**
   * @brief Construct field with component-wise initialization.
   *
   * @tparam Args Types of the component values.
   * @param args Individual values for each component.
   */
  template <typename... Args,
            typename = std::enable_if_t<sizeof...(Args) == components> >
  KOKKOS_FORCEINLINE_FUNCTION constexpr field(Args &&...args)
      : m_data(std::forward<Args>(args)...) {}

  /**
   * @brief Access field component by index (read-only).
   *
   * @param icomp Component index (0 to components-1).
   * @return Const reference to the component value.
   */
  KOKKOS_FORCEINLINE_FUNCTION const typename value_type::value_type &
  operator()(const std::size_t icomp) const {
    return m_data(icomp);
  }

  /**
   * @brief Access field component by index (read-write).
   *
   * @param icomp Component index (0 to components-1).
   * @return Mutable reference to the component value.
   */
  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type &
  operator()(const std::size_t icomp) {
    return m_data(icomp);
  }

  /**
   * @brief Equality comparison operator.
   *
   * @param other The field object to compare against.
   * @return True if all components are equal.
   */
  KOKKOS_FORCEINLINE_FUNCTION bool operator==(const field &other) const {
    return (this->m_data == other.m_data);
  }

  /**
   * @brief Inequality comparison operator.
   *
   * @param other The field object to compare against.
   * @return True if any component differs.
   */
  KOKKOS_FORCEINLINE_FUNCTION bool operator!=(const field &other) const {
    return !(*this == other);
  }

  /**
   * @brief Multiplication assignment operator.
   *
   * @param other The scalar factor to multiply with.
   * @return Reference to this field.
   */
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto &
  operator*=(const typename value_type::value_type &other) {
    this->m_data *= other;
    return *this;
  }

  /**
   * @brief Convert field to string representation.
   *
   * @return String containing all component values.
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
