#pragma once

#include "simd.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {
/**
 * @brief Datatype used to scalar values within chunk of elements. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the scalar values
 * @tparam NumberOfElements Number of elements in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, specfem::dimension::type DimensionTag,
          int NumberOfElements, int NumberOfGLLPoints, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct ScalarChunkViewType;

template <typename T, int NumberOfElements, int NumberOfGLLPoints, bool UseSIMD,
          typename MemorySpace, typename MemoryTraits>
struct ScalarChunkViewType<T, specfem::dimension::type::dim2, NumberOfElements,
                           NumberOfGLLPoints, UseSIMD, MemorySpace,
                           MemoryTraits>
    : public Kokkos::View<
          typename specfem::datatype::simd<T, UseSIMD>::datatype
              [NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints],
          MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type =
      Kokkos::View<typename specfem::datatype::simd<T, UseSIMD>::datatype
                       [NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints],
                   MemorySpace, MemoryTraits>; ///< Underlying data type used to
                                               ///< store values
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  constexpr static bool using_simd = UseSIMD;   ///< Use SIMD datatypes for the
                                                ///< array. If false,
                                                ///< std::is_same<value_type,
                                                ///< base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int nelements = NumberOfElements; ///< Number of elements in
                                                     ///< the chunk
  constexpr static int ngll = NumberOfGLLPoints; ///< Number of GLL points in
                                                 ///< each element
  constexpr static bool isChunkViewType = true;
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
  ScalarChunkViewType() = default;

  /**
   * @brief Construct a new ScalarChunkViewType object within
   * ScratchMemorySpace.
   * Allocates an unmanaged view within ScratchMemorySpace. Useful for
   * generating scratch views.
   *
   * @tparam ScratchMemorySpace Memory space of the view
   * @param scratch_memory_space Memory space of the view
   */
  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION
  ScalarChunkViewType(const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<
            typename specfem::datatype::simd<T, UseSIMD>::datatype
                [NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints],
            MemorySpace, MemoryTraits>(scratch_memory_space) {}
  ///@}
};

/**
 * @brief Datatype used to vector values within chunk of elements. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the vector values
 * @tparam NumberOfElements Number of elements in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam Components Number of vector values (components) at each GLL point
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <
    typename T, specfem::dimension::type DimensionTag, int NumberOfElements,
    int NumberOfGLLPoints, int Components, bool UseSIMD = false,
    typename MemorySpace = Kokkos::DefaultExecutionSpace::scratch_memory_space,
    typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct VectorChunkViewType;

template <typename T, int NumberOfElements, int NumberOfGLLPoints,
          int Components, bool UseSIMD, typename MemorySpace,
          typename MemoryTraits>
struct VectorChunkViewType<T, specfem::dimension::type::dim2, NumberOfElements,
                           NumberOfGLLPoints, Components, UseSIMD, MemorySpace,
                           MemoryTraits>
    : public Kokkos::View<typename specfem::datatype::simd<T, UseSIMD>::datatype
                              [NumberOfElements][NumberOfGLLPoints]
                              [NumberOfGLLPoints][Components],
                          MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type =
      Kokkos::View<typename simd::datatype[NumberOfElements][NumberOfGLLPoints]
                                          [NumberOfGLLPoints][Components],
                   MemorySpace, MemoryTraits>; ///< Underlying data type used to
                                               ///< store values
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  constexpr static bool using_simd = UseSIMD;   ///< Use SIMD datatypes for the
                                                ///< array. If false,
                                                ///< std::is_same<value_type,
                                                ///< base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int nelements = NumberOfElements; ///< Number of elements in
                                                     ///< the chunk
  constexpr static int ngll = NumberOfGLLPoints; ///< Number of GLL points in
                                                 ///< each element
  constexpr static int components = Components;  ///< Number of vector values at
                                                 ///< each GLL point
  constexpr static bool isChunkViewType = true;
  constexpr static bool isScalarViewType = true;
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
  VectorChunkViewType() = default;

  /**
   * @brief Construct a new VectorChunkViewType object within
   * ScratchMemorySpace.
   * Allocates an unmanaged view within ScratchMemorySpace. Useful for
   * generating scratch views.
   *
   * @tparam ScratchMemorySpace Memory space of the view
   * @param scratch_memory_space Memory space of the view
   */
  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION
  VectorChunkViewType(const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<value_type[NumberOfElements][NumberOfGLLPoints]
                               [NumberOfGLLPoints][Components],
                     MemorySpace, MemoryTraits>(scratch_memory_space) {}
  ///@}
};

/**
 * @brief Datatype used to tensor values within chunk of elements. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the tensor values
 * @tparam NumberOfElements Number of elements in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam Components Number of vector values (components) at each GLL point
 * @tparam NumberOfDimensions Number of dimensions of the tensor
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, specfem::dimension::type DimensionTag,
          int NumberOfElements, int NumberOfGLLPoints, int Components,
          int NumberOfDimensions, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct TensorChunkViewType;

template <typename T, int NumberOfElements, int NumberOfGLLPoints,
          int Components, int NumberOfDimensions, bool UseSIMD,
          typename MemorySpace, typename MemoryTraits>
struct TensorChunkViewType<T, specfem::dimension::type::dim2, NumberOfElements,
                           NumberOfGLLPoints, Components, NumberOfDimensions,
                           UseSIMD, MemorySpace, MemoryTraits>
    : public Kokkos::View<
          typename specfem::datatype::simd<T, UseSIMD>::datatype
              [NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints]
              [Components][NumberOfDimensions],
          MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type = typename Kokkos::View<
      typename simd::datatype[NumberOfElements][NumberOfGLLPoints]
                             [NumberOfGLLPoints][Components]
                             [NumberOfDimensions],
      MemorySpace, MemoryTraits>; ///< Underlying data type used to store values
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  constexpr static bool using_simd = UseSIMD;   ///< Use SIMD datatypes for the
                                                ///< array. If false,
                                                ///< std::is_same<value_type,
                                                ///< base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int nelements = NumberOfElements; ///< Number of elements in
                                                     ///< the chunk
  constexpr static int ngll = NumberOfGLLPoints; ///< Number of GLL points in
                                                 ///< each element
  constexpr static int components = Components;  ///< Number of tensor values at
                                                 ///< each GLL point
  constexpr static int dimensions =
      NumberOfDimensions; ///< Number of dimensions
                          ///< of the tensor values
  constexpr static bool isChunkViewType = true;
  constexpr static bool isScalarViewType = false;
  ///@}

  /**
   * @name Constructors and assignment operators
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  TensorChunkViewType() = default;

  /**
   * @brief Construct a new TensorChunkViewType object within
   * ScratchMemorySpace.
   * Allocates an unmanaged view within ScratchMemorySpace. Useful for
   * generating scratch views.
   *
   * @tparam ScratchMemorySpace Memory space of the view
   * @param scratch_memory_space Memory space of the view
   */
  template <typename ScratchMemorySpace,
            typename std::enable_if<
                std::is_same<MemorySpace, ScratchMemorySpace>::value,
                bool>::type = true>
  KOKKOS_FUNCTION
  TensorChunkViewType(const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<
            value_type[NumberOfElements][NumberOfGLLPoints][NumberOfGLLPoints]
                      [Components][NumberOfDimensions],
            MemorySpace, MemoryTraits>(scratch_memory_space) {}
};

} // namespace datatype
} // namespace specfem
