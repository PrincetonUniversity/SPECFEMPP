#pragma once

#include "specfem/data_access.hpp"
#include "specfem/element.hpp"
#include "specfem/setup.hpp"

namespace specfem {
namespace point {

/**
 * @brief Struct to store the index associated with a quadrature point
 *
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 * @tparam using_simd Flag to indicate if this is a simd index
 */
template <specfem::element::dimension_tag DimensionTag, bool using_simd = false>
struct index;

//--------------------------- 2D Specializations -----------------------------//

/**
 * @brief 2D specialization of the index struct for the non-SIMD case
 *
 */
template <>
struct index<specfem::element::dimension_tag::dim2, false>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::index,
          specfem::element::dimension_tag::dim2, false> {
  /**
   * @brief Index of the spectral element.
   */
  int ispec;

  /**
   * @brief Index of the quadrature point in the z direction within the spectral
   * element.
   */
  int iz;

  /**
   * @brief Index of the quadrature point in the x direction within the spectral
   * element.
   */
  int ix;

  /**
   * @brief Default constructor.
   */
  KOKKOS_FUNCTION
  index() = default;

  /**
   * @brief Construct a new index object.
   *
   * @param ispec Index of the spectral element.
   * @param iz Index of the quadrature point in the z direction within the
   * spectral element.
   * @param ix Index of the quadrature point in the x direction within the
   * spectral element.
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const int &iz, const int &ix)
      : ispec(ispec), iz(iz), ix(ix) {}

  /**
   * @brief Copy assignment operator.
   *
   * @param other The index to copy from.
   * @return Reference to this index.
   */
  KOKKOS_FUNCTION
  index &operator=(const index &other) {
    if (this != &other) {
      ispec = other.ispec;
      iz = other.iz;
      ix = other.ix;
    }
    return *this;
  }
};

/**
 * @brief 2D specialization of the index struct for the SIMD case
 *
 */
template <>
struct index<specfem::element::dimension_tag::dim2, true>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::index,
          specfem::element::dimension_tag::dim2, true> {
  /**
   * @brief Index associated with the spectral element at the start of the SIMD
   * vector.
   */
  int ispec;

  /**
   * @brief Number of elements stored in the SIMD vector.
   */
  int number_elements;

  /**
   * @brief Index of the quadrature point in the z direction within the spectral
   * element.
   */
  int iz;

  /**
   * @brief Index of the quadrature point in the x direction within the spectral
   * element.
   */
  int ix;

  /**
   * @brief Default constructor.
   */
  KOKKOS_FUNCTION
  index() = default;

  /**
   * @brief Construct a new simd index object.
   *
   * @param ispec Index of the spectral element.
   * @param number_elements Number of elements.
   * @param iz Index of the quadrature point in the z direction within the
   * spectral element.
   * @param ix Index of the quadrature point in the x direction within the
   * spectral element.
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const int &number_elements, const int &iz,
        const int &ix)
      : ispec(ispec), number_elements(number_elements), iz(iz), ix(ix) {}

  /**
   * @brief Copy assignment operator.
   *
   * @param other The index to copy from.
   * @return Reference to this index.
   */
  KOKKOS_FUNCTION
  index &operator=(const index &other) {
    if (this != &other) {
      ispec = other.ispec;
      number_elements = other.number_elements;
      iz = other.iz;
      ix = other.ix;
    }
    return *this;
  }

  /**
   * @brief Returns a boolean mask to check if the SIMD index is within the SIMD
   * vector.
   *
   * @param lane SIMD lane.
   * @return bool True if the SIMD index is within the SIMD vector.
   */
  KOKKOS_INLINE_FUNCTION
  bool mask(const std::size_t &lane) const {
    return int(lane) < number_elements;
  }
};

//-------------------------- 3D Specializations ------------------------------//

/**
 * @brief Template specialization for 3D elements for the non-SIMD index
 *        implementation.
 *
 */
template <>
struct index<specfem::element::dimension_tag::dim3, false>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::index,
          specfem::element::dimension_tag::dim3, false> {
  /**
   * @brief Index of the spectral element.
   */
  int ispec;

  /**
   * @brief Index of the quadrature point in the z direction within the spectral
   * element.
   */
  int iz;

  /**
   * @brief Index of the quadrature point in the y direction within the spectral
   * element.
   */
  int iy;

  /**
   * @brief Index of the quadrature point in the x direction within the spectral
   * element.
   */
  int ix;

  /**
   * @brief Default constructor.
   */
  KOKKOS_FUNCTION
  index() = default;

  /**
   * @brief Construct a new index object.
   *
   * @param ispec Index of the spectral element.
   * @param iz Index of the quadrature point in the z direction within the
   * spectral element.
   * @param iy Index of the quadrature point in the y direction within the
   * spectral element.
   * @param ix Index of the quadrature point in the x direction within the
   * spectral element.
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const int &iz, const int &iy, const int &ix)
      : ispec(ispec), iz(iz), iy(iy), ix(ix) {};

  /**
   * @brief Copy assignment operator.
   *
   * @param other The index to copy from.
   * @return Reference to this index.
   */
  KOKKOS_FUNCTION
  index &operator=(const index &other) {
    if (this != &other) {
      ispec = other.ispec;
      iz = other.iz;
      iy = other.iy;
      ix = other.ix;
    }
    return *this;
  }
};

/**
 * @brief Template specialization for 2D elements for the SIMD index
 * implementation.
 *
 */
template <>
struct index<specfem::element::dimension_tag::dim3, true>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::index,
          specfem::element::dimension_tag::dim3, true> {
  /**
   * @brief Index associated with the spectral element at the start of the SIMD
   * vector.
   */
  int ispec;

  /**
   * @brief Number of elements stored in the SIMD vector.
   */
  int number_elements;

  /**
   * @brief Index of the quadrature point in the z direction within the spectral
   * element.
   */
  int iz;

  /**
   * @brief Index of the quadrature point in the y direction within the spectral
   * element.
   */
  int iy;

  /**
   * @brief Index of the quadrature point in the x direction within the spectral
   * element.
   */
  int ix;

  /**
   * @brief Default constructor.
   */
  KOKKOS_FUNCTION
  index() = default;

  /**
   * @brief Construct a new simd index object.
   *
   * @param ispec Index of the spectral element.
   * @param number_elements Number of elements.
   * @param iz Index of the quadrature point in the z direction within the
   * spectral element.
   * @param iy Index of the quadrature point in the y direction within the
   * spectral element.
   * @param ix Index of the quadrature point in the x direction within the
   * spectral element.
   */
  KOKKOS_FUNCTION
  index(const int &ispec, const int &number_elements, const int &iz,
        const int &iy, const int &ix)
      : ispec(ispec), number_elements(number_elements), iz(iz), iy(iy), ix(ix) {
  }

  /**
   * @brief Copy assignment operator.
   *
   * @param other The index to copy from.
   * @return Reference to this index.
   */
  KOKKOS_FUNCTION
  index &operator=(const index &other) {
    if (this != &other) {
      ispec = other.ispec;
      number_elements = other.number_elements;
      iz = other.iz;
      iy = other.iy;
      ix = other.ix;
    }
    return *this;
  }

  /**
   * @brief Returns a boolean mask to check if the SIMD index is within the SIMD
   * vector.
   *
   * @param lane SIMD lane.
   * @return bool True if the SIMD index is within the SIMD vector.
   */
  KOKKOS_INLINE_FUNCTION
  bool mask(const std::size_t &lane) const {
    return int(lane) < number_elements;
  }
};

} // namespace point
} // namespace specfem
