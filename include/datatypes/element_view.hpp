#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

/**
 * @brief Datatype used to scalar values within an element. Data is stored
 * within a Kokkos view located in the memory space specified by MemorySpace.
 *
 * @tparam T Data type of the scalar values
 * @tparam NumberOfGLLPoints Number of GLL points in the element
 * @tparam Components Number of scalar values (components) at each GLL point
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, int NumberOfGLLPoints, int Components,
          typename MemorySpace, typename MemoryTraits, bool UseSIMD = false>
struct ScalarElementViewType
    : public Kokkos::View<
          typename specfem::datatype::simd<T, UseSIMD>::datatype
              [NumberOfGLLPoints][NumberOfGLLPoints][Components],
          Kokkos::LayoutLeft, MemorySpace, MemoryTraits> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type = Kokkos::View<
      typename simd::datatype[NumberOfGLLPoints][NumberOfGLLPoints][Components],
      Kokkos::LayoutLeft, MemorySpace, MemoryTraits>; ///< Underlying data type
                                                      ///< used to store values
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  constexpr static bool using_simd =
      UseSIMD; ///< Use SIMD datatypes for the array. If false,
               ///< std::is_same<value_type, base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int ngll = NumberOfGLLPoints; ///< Number of GLL points in
                                                 ///< the element
  constexpr static int components = Components;  ///< Number of scalar values at
                                                 ///< each GLL point
  constexpr static bool isPointViewType = false;
  constexpr static bool isElementViewType = true;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = true;
  constexpr static bool isVectorViewType = false;
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
   * ScratchMemorySpace
   *
   * Allocates an unmanaged view within ScratchMemorySpace. Useful for
   * generating scratch views.
   *
   * @tparam ScratchMemorySpace Memory space of the view
   * @param scratch_space Scratch memory space
   */
  template <typename ScratchMemorySpace,
            typename std::enable_if<
                std::is_same<MemorySpace, ScratchMemorySpace>::value,
                bool>::type = true>
  KOKKOS_FUNCTION ScalarElementViewType(const ScratchMemorySpace &scratch_space)
      : Kokkos::View<
            value_type[NumberOfGLLPoints][NumberOfGLLPoints][Components],
            Kokkos::LayoutLeft, MemorySpace, MemoryTraits>(scratch_space) {}
  ///@}
};

/**
 * @brief Datatype used to store vector values within an element. Data is stored
 * within a Kokkos view located in the memory space specified by MemorySpace.
 *
 * @tparam T Data type of the vector values
 * @tparam NumberOfGLLPoints Number of GLL points in the element
 * @tparam Components Number of vector values (components) at each GLL point
 * @tparam NumberOfDimensions Number of dimensions of the vector values
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, int NumberOfGLLPoints, int Components,
          int NumberOfDimensions, typename MemorySpace, typename MemoryTraits,
          bool UseSIMD = false>
struct VectorElementViewType
    : public Kokkos::View<typename specfem::datatype::simd<T, UseSIMD>::datatype
                              [NumberOfGLLPoints][NumberOfGLLPoints]
                              [NumberOfDimensions][Components],
                          Kokkos::LayoutLeft, MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type =
      Kokkos::View<typename simd::datatype[NumberOfGLLPoints][NumberOfGLLPoints]
                                          [NumberOfDimensions][Components],
                   Kokkos::LayoutLeft, MemorySpace,
                   MemoryTraits>; ///< Underlying data type used to store values
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
  constexpr static int ngll = NumberOfGLLPoints; ///< Number of GLL points in
                                                 ///< the element
  constexpr static int components = Components;  ///< Number of vector values at
                                                 ///< each GLL point
  constexpr static int dimensions =
      NumberOfDimensions; ///< Number of dimensions
                          ///< of the vector values
  constexpr static bool isPointViewType = false;
  constexpr static bool isElementViewType = true;
  constexpr static bool isChunkViewType = false;
  constexpr static bool isDomainViewType = false;
  constexpr static bool isScalarViewType = false;
  constexpr static bool isVectorViewType = true;
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
  VectorElementViewType() = default;

  /**
   * @brief Construct a new VectorElementViewType object within
   * ScratchMemorySpace
   *
   * Allocates an unmanaged view within ScratchMemorySpace. Useful for
   * generating scratch views.
   *
   * @tparam ScratchMemorySpace Memory space of the view
   * @param scratch_space Scratch memory space
   */
  template <typename ScratchMemorySpace,
            typename std::enable_if<
                std::is_same<MemorySpace, ScratchMemorySpace>::value,
                bool>::type = true>
  KOKKOS_FUNCTION VectorElementViewType(const ScratchMemorySpace &scratch_space)
      : Kokkos::View<value_type[NumberOfGLLPoints][NumberOfGLLPoints]
                               [NumberOfDimensions][Components],
                     Kokkos::LayoutLeft, MemorySpace, MemoryTraits>(
            scratch_space) {}
  ///@}
};

} // namespace datatype
} // namespace specfem
