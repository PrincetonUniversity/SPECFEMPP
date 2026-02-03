#pragma once

#include "enumerations/interface.hpp"
#include "impl/chunk_edge_subview.hpp"
#include "simd.hpp"
#include <Kokkos_Core.hpp>

// Forward declarations
namespace specfem::point {
template <specfem::dimension::type DimensionTag, bool UseSIMD> struct index;
} // namespace specfem::point

namespace specfem {
namespace datatype {

/**
 * @brief Datatype used to scalar values within chunk of edges. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the scalar values
 * @tparam NumberOfEdges Number of edges in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, specfem::dimension::type DimensionTag, int NumberOfEdges,
          int NumberOfGLLPoints, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct ScalarChunkEdgeViewType;

/**
 * @brief 2D specialization for ScalarChunkEdgeViewType
 *
 * @tparam T Data type of the scalar values
 * @tparam NumberOfEdges Number of edges in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 */
template <typename T, int NumberOfEdges, int NumberOfGLLPoints, bool UseSIMD,
          typename MemorySpace, typename MemoryTraits>
struct ScalarChunkEdgeViewType<T, specfem::dimension::type::dim2, NumberOfEdges,
                               NumberOfGLLPoints, UseSIMD, MemorySpace,
                               MemoryTraits>
    : public Kokkos::View<typename specfem::datatype::simd<T, UseSIMD>::datatype
                              [NumberOfEdges][NumberOfGLLPoints],
                          MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type =
      Kokkos::View<typename simd::datatype[NumberOfEdges][NumberOfGLLPoints],
                   MemorySpace, MemoryTraits>; ///< Underlying data type used to
                                               ///< store values
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  using index_type =
      typename specfem::point::index<specfem::dimension::type::dim2,
                                     UseSIMD>; ///< index type for accessing at
                                               ///< GLL level
  constexpr static bool using_simd = UseSIMD;  ///< Use SIMD datatypes for the
                                               ///< array. If false,
                                               ///< std::is_same<value_type,
                                               ///< base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge; ///< Accessor type for
                                                      ///< identifying the
                                                      ///< class

  constexpr static int nedges = NumberOfEdges;   ///< Number of edges in
                                                 ///< the chunk
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
  ScalarChunkEdgeViewType() = default;

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
  ScalarChunkEdgeViewType(const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<value_type[NumberOfEdges][NumberOfGLLPoints], MemorySpace,
                     MemoryTraits>(scratch_memory_space) {}
  ///@}

  using type::operator();

  /**
   * @brief Get scalar value by a point index.
   *
   * @param index Point index
   */
  KOKKOS_INLINE_FUNCTION
  constexpr value_type &operator()(index_type index) {
    return (*this)(index.ispec, index.ipoint);
  }
};

/**
 * @brief Datatype used to vector values within chunk of edges. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the vector values
 * @tparam NumberOfEdges Number of edges in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam Components Number of vector values (components) at each GLL point
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, specfem::dimension::type DimensionTag, int NumberOfEdges,
          int NumberOfGLLPoints, int Components, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct VectorChunkEdgeViewType;

/**
 * @brief 2D specialization for VectorChunkEdgeViewType
 *
 * @tparam T Data type of the vector values
 * @tparam NumberOfEdges Number of edges in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam Components Number of vector values (components) at each GLL point
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 */
template <typename T, int NumberOfEdges, int NumberOfGLLPoints, int Components,
          bool UseSIMD, typename MemorySpace, typename MemoryTraits>
struct VectorChunkEdgeViewType<T, specfem::dimension::type::dim2, NumberOfEdges,
                               NumberOfGLLPoints, Components, UseSIMD,
                               MemorySpace, MemoryTraits>
    : public Kokkos::View<typename specfem::datatype::simd<T, UseSIMD>::datatype
                              [NumberOfEdges][NumberOfGLLPoints][Components],
                          MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type = Kokkos::View<
      typename simd::datatype[NumberOfEdges][NumberOfGLLPoints][Components],
      MemorySpace, MemoryTraits>; ///< Underlying data type used to
                                  ///< store values
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  using index_type =
      typename specfem::point::index<specfem::dimension::type::dim2,
                                     UseSIMD>; ///< index type for accessing at
                                               ///< GLL level
  constexpr static bool using_simd = UseSIMD;  ///< Use SIMD datatypes for the
                                               ///< array. If false,
                                               ///< std::is_same<value_type,
                                               ///< base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge; ///< Accessor type for
                                                      ///< identifying the
                                                      ///< class
  constexpr static int nedges = NumberOfEdges;        ///< Number of edges in
                                                      ///< the chunk
  constexpr static int ngll = NumberOfGLLPoints; ///< Number of GLL points in
                                                 ///< each element
  constexpr static int components = Components;  ///< Number of vector values at
                                                 ///< each GLL point
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
  VectorChunkEdgeViewType() = default;

  /**
   * @brief Construct a new VectorChunkEdgeViewType object within
   * ScratchMemorySpace.
   * Allocates an unmanaged view within ScratchMemorySpace. Useful for
   * generating scratch views.
   *
   * @tparam ScratchMemorySpace Memory space of the view
   * @param scratch_memory_space Memory space of the view
   */
  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION
  VectorChunkEdgeViewType(const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<value_type[NumberOfEdges][NumberOfGLLPoints][Components],
                     MemorySpace, MemoryTraits>(scratch_memory_space) {}
  ///@}

  using type::operator();

  /**
   * @brief Get vector component by a point index and vector indices.
   *
   * @param index Point index
   * @param icomp Component index
   */
  KOKKOS_INLINE_FUNCTION
  constexpr value_type &operator()(const index_type &index, const int &icomp) {
    return (*this)(index.ispec, index.ipoint, icomp);
  }

  /**
   * @brief Get vector subview by a point index.
   *
   * @param index Point index
   */
  KOKKOS_INLINE_FUNCTION
  impl::VectorChunkEdgeSubview<VectorChunkEdgeViewType>
  operator()(const index_type &index) {
    return { *this, index };
  }
};

/**
 * @brief Datatype used to tensor values within chunk of edges. Data is
 * stored within a Kokkos view located in the memory space specified by
 * MemorySpace.
 *
 * @tparam T Data type of the tensor values
 * @tparam NumberOfEdges Number of edges in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam Components Number of vector values (components) at each GLL point
 * @tparam NumberOfDimensions Number of dimensions of the tensor
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, specfem::dimension::type DimensionTag, int NumberOfEdges,
          int NumberOfGLLPoints, int Components, int NumberOfDimensions,
          bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct TensorChunkEdgeViewType;

/**
 * @brief 2D specialization for TensorChunkEdgeViewType
 *
 * @tparam T Data type of the tensor values
 * @tparam NumberOfEdges Number of edges in the chunk
 * @tparam NumberOfGLLPoints Number of GLL points in each element
 * @tparam Components Number of vector values (components) at each GLL point
 * @tparam NumberOfDimensions Number of dimensions of the tensor
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 * @tparam MemorySpace Memory space of the view
 * @tparam MemoryTraits Memory traits of the view
 */
template <typename T, int NumberOfEdges, int NumberOfGLLPoints, int Components,
          int NumberOfDimensions, bool UseSIMD, typename MemorySpace,
          typename MemoryTraits>
struct TensorChunkEdgeViewType<
    T, specfem::dimension::type::dim2, NumberOfEdges, NumberOfGLLPoints,
    Components, NumberOfDimensions, UseSIMD, MemorySpace, MemoryTraits>
    : public Kokkos::View<typename specfem::datatype::simd<T, UseSIMD>::datatype
                              [NumberOfEdges][NumberOfGLLPoints][Components]
                              [NumberOfDimensions],
                          MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type = typename Kokkos::View<
      typename simd::datatype[NumberOfEdges][NumberOfGLLPoints][Components]
                             [NumberOfDimensions],
      MemorySpace, MemoryTraits>; ///< Underlying data type used to store values
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  using index_type =
      typename specfem::point::index<specfem::dimension::type::dim2,
                                     UseSIMD>; ///< index type for accessing at
                                               ///< GLL level
  constexpr static bool using_simd = UseSIMD;  ///< Use SIMD datatypes for the
                                               ///< array. If false,
                                               ///< std::is_same<value_type,
                                               ///< base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge; ///< Accessor type for
                                                      ///< identifying the
                                                      ///< class

  constexpr static int nedges = NumberOfEdges;   ///< Number of edges in
                                                 ///< the chunk
  constexpr static int ngll = NumberOfGLLPoints; ///< Number of GLL points in
                                                 ///< each element
  constexpr static int components = Components;  ///< Number of tensor values at
                                                 ///< each GLL point
  constexpr static int dimensions =
      NumberOfDimensions; ///< Number of dimensions
                          ///< of the tensor values
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
  TensorChunkEdgeViewType() = default;

  /**
   * @brief Construct a new TensorChunkEdgeViewType object within
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
  TensorChunkEdgeViewType(const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<value_type[NumberOfEdges][NumberOfGLLPoints][Components]
                               [NumberOfDimensions],
                     MemorySpace, MemoryTraits>(scratch_memory_space) {}
  ///@}

  using type::operator();

  /**
   * @brief Get tensor component by a point index and tensor indices.
   *
   * @param index Point index
   * @param icomp Component index
   * @param idim Dimension index
   */
  KOKKOS_INLINE_FUNCTION
  constexpr value_type &operator()(const index_type &index, const int &icomp,
                                   const int &idim) {
    return (*this)(index.ispec, index.ipoint, icomp, idim);
  }

  /**
   * @brief Get tensor subview by a point index.
   *
   * @param index Point index
   */
  KOKKOS_INLINE_FUNCTION
  impl::TensorChunkEdgeSubview<TensorChunkEdgeViewType>
  operator()(const index_type &index) {
    return { *this, index };
  }
};

} // namespace datatype
} // namespace specfem
