#pragma once

#include "enumerations/interface.hpp"
#include "point_view.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

/**
 * @brief Datatype used to scalar values within an element. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the scalar values
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 */
template <
    typename T, specfem::dimension::type DimensionTag, int NumberOfGLLPoints,
    typename MemorySpace = Kokkos::DefaultExecutionSpace::scratch_memory_space,
    typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct ScalarElementViewType;

/* 2D specialization */
template <typename T, int NumberOfGLLPoints, typename MemorySpace,
          typename MemoryTraits>
struct ScalarElementViewType<T, specfem::dimension::type::dim2,
                             NumberOfGLLPoints, MemorySpace, MemoryTraits>
    : public Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints],
                          Kokkos::LayoutRight, MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using type =
      Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints], Kokkos::LayoutRight,
                   MemorySpace, MemoryTraits>; ///< Underlying data type
                                               ///< used to store values
  using value_type = T;                        ///< Value type used to store
                                               ///< the elements of the array
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::chunk_element; ///< Accessor type for
                                                         ///< identifying the
                                                         ///< class

  constexpr static int ngll = NumberOfGLLPoints; ///< Number of GLL points in
                                                 ///< each element
  ///@}

  /**
   * @name Constructors and assignment operators
   *
   */
  ///@{
  /**
   * @brief Default constructor
   */
  KOKKOS_FUNCTION
  ScalarElementViewType() = default;

  /**
   * @brief Construct a new ScalarElementViewType object within
   * ScratchMemorySpace.
   * Allocates an unmanaged view within ScratchMemorySpace. Useful for
   * generating scratch views.
   *
   * @tparam ScratchMemorySpace Memory space of the view
   * @param scratch_memory_space Memory space of the view
   */
  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION
  ScalarElementViewType(const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<value_type[NumberOfGLLPoints][NumberOfGLLPoints],
                     Kokkos::LayoutRight, MemorySpace, MemoryTraits>(
            scratch_memory_space) {}
  ///@}

  using type::operator();

  /**
   * @brief Get scalar value by a point index.
   *
   * @param index Point index
   */
  template <typename IndexType>
  KOKKOS_INLINE_FUNCTION constexpr value_type &
  operator()(const IndexType &index) {
    return (*this)(index.iz, index.ix);
  }
};

/* 3D specialization */
template <typename T, int NumberOfGLLPoints, typename MemorySpace,
          typename MemoryTraits>
struct ScalarElementViewType<T, specfem::dimension::type::dim3,
                             NumberOfGLLPoints, MemorySpace, MemoryTraits>
    : public Kokkos::View<
          T[NumberOfGLLPoints][NumberOfGLLPoints][NumberOfGLLPoints],
          Kokkos::LayoutRight, MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using type =
      Kokkos::View<T[NumberOfGLLPoints][NumberOfGLLPoints][NumberOfGLLPoints],
                   Kokkos::LayoutRight, MemorySpace,
                   MemoryTraits>; ///< Underlying
                                  ///< data
                                  ///< type
                                  ///< used to
                                  ///< store
                                  ///< values
  using value_type = T;           ///< Value type used to store
                                  ///< the elements of the array
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::chunk_element; ///< Accessor type for
                                                         ///< identifying the
                                                         ///< class

  constexpr static int ngll = NumberOfGLLPoints; ///< Number of GLL points in
                                                 ///< each element
  ///@}

  /**
   * @name Constructors and assignment operators
   *
   */
  ///@{
  /**
   * @brief Default constructor
   */
  KOKKOS_FUNCTION
  ScalarElementViewType() = default;

  /**
   * @brief Construct a new ScalarElementViewType object within
   * ScratchMemorySpace.
   * Allocates an unmanaged view within ScratchMemorySpace. Useful for
   * generating scratch views.
   *
   * @tparam ScratchMemorySpace Memory space of the view
   * @param scratch_memory_space Memory space of the view
   */
  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION
  ScalarElementViewType(const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<
            value_type[NumberOfGLLPoints][NumberOfGLLPoints][NumberOfGLLPoints],
            Kokkos::LayoutRight, MemorySpace, MemoryTraits>(
            scratch_memory_space) {}
  ///@}

  using type::operator();

  /**
   * @brief Get scalar value by a point index.
   *
   * @param index Point index
   */
  template <typename IndexType>
  KOKKOS_INLINE_FUNCTION constexpr value_type &
  operator()(const IndexType &index) {
    return (*this)(index.iz, index.iy, index.ix);
  }
};

} // namespace datatype
} // namespace specfem
