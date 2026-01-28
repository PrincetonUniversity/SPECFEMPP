#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/macros.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_SIMD.hpp>
#include <boost/preprocessor.hpp>
#include <iostream>
#include <sstream>

namespace specfem::point::impl {

namespace properties {

/**
 * @brief Compile time information associated with the properties of a
 * quadrature point in a 2D
 *
 * @tparam Dimension The dimension of the medium
 * @tparam MediumTag The type of the medium
 * @tparam PropertyTag The type of the properties
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct PropertyAccessor : public specfem::data_access::Accessor<
                              specfem::data_access::AccessorType::point,
                              specfem::data_access::DataClassType::properties,
                              DimensionTag, UseSIMD> {

public:
  using base_accessor = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::properties, DimensionTag,
      UseSIMD>; ///< Base type of
                ///< the point
                ///< properties

  using simd =
      typename base_accessor::template simd<type_real>; ///< SIMD data type

  using value_type =
      typename base_accessor::template scalar_type<type_real>; ///< Type of the
                                                               ///< properties

  constexpr static auto medium_tag = MediumTag;     ///< type of the medium
  constexpr static auto property_tag = PropertyTag; ///< type of the properties
};

/**
 * @brief Data container to hold properties of a medium at a quadrature point
 *
 * @tparam DimensionTag The dimension of the medium
 * @tparam MediumTag The type of the medium
 * @tparam PropertyTag The type of the properties
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 * @tparam Enable SFINAE enable parameter
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct data_container;
} // namespace properties

namespace kernels {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct KernelsAccessor
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::kernels, DimensionTag, UseSIMD> {
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::kernels, DimensionTag,
      UseSIMD>;                                              ///< Base type of
                                                             ///< the point
                                                             ///< kernels
  using simd = typename base_type::template simd<type_real>; ///< SIMD data type
  using value_type =
      typename base_type::template scalar_type<type_real>; ///< Type of the
                                                           ///< properties

  constexpr static auto medium_tag = MediumTag;     ///< type of the medium
  constexpr static auto property_tag = PropertyTag; ///< type of the properties
};

/**
 * @brief Data container to hold kernels at a quadrature point
 *
 * @tparam DimensionTag The dimension of the medium
 * @tparam MediumTag The type of the medium
 * @tparam PropertyTag The type of the kernels
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 * @tparam Enable SFINAE enable parameter
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct data_container;
} // namespace kernels

} // namespace specfem::point::impl
