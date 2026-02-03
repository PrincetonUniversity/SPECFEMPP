#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/datatype.hpp"

namespace specfem::data_access {

/**
 * @brief Point-wise accessor for single quadrature point operations.
 *
 * Provides SIMD-optimized data access for individual quadrature point
 * computations. Uses specialized view types for efficient point-wise
 * data operations with configurable vectorization.
 *
 * @tparam DataClass Type of point data (properties, fields, etc.)
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct Accessor<specfem::data_access::AccessorType::point, DataClass,
                DimensionTag, UseSIMD> {
  /// @brief Accessor pattern identifier
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::point;
  /// @brief Data classification type
  constexpr static auto data_class = DataClass;
  /// @brief Spatial dimension
  constexpr static auto dimension_tag = DimensionTag;
  /// @brief SIMD vectorization flag
  constexpr static bool using_simd = UseSIMD;

  /**
   * @brief Datatype used to store data with optional SIMD vectorization
   *
   * @tparam T Base data type
   */
  template <typename T> using simd = specfem::datatype::simd<T, UseSIMD>;

  /**
   * @brief Scalar field storage for single point
   *
   * @tparam T Base data type
   */
  template <typename T> using scalar_type = typename simd<T>::datatype;

  /**
   * @brief Vector field storage for single point
   *
   * @tparam T Base data type
   * @tparam dimension Vector dimension (2D/3D)
   */
  template <typename T, int dimension>
  using vector_type =
      typename specfem::datatype::VectorPointViewType<T, dimension, UseSIMD>;

  /**
   * @brief Tensor field storage for single point
   *
   * @tparam T Base data type
   * @tparam components Number of tensor components
   * @tparam dimension Spatial dimension
   */
  template <typename T, int components, int dimension>
  using tensor_type =
      typename specfem::datatype::TensorPointViewType<T, components, dimension,
                                                      UseSIMD>;
};

/**
 * @brief Type trait to detect point accessor types.
 */
template <typename T, typename = void> struct is_point : std::false_type {};

template <typename T>
struct is_point<T, std::enable_if_t<T::accessor_type ==
                                    specfem::data_access::AccessorType::point> >
    : std::true_type {};

} // namespace specfem::data_access
