#pragma once

#include "accessor_type.hpp"
#include "simd.hpp"
#include "specfem/element/tags.hpp"
#include <Kokkos_Core.hpp>
#include <utility>

// Forward declarations
namespace specfem::point {
template <specfem::element::dimension_tag DimensionTag, bool UseSIMD>
struct index;
} // namespace specfem::point

namespace specfem::datatype {

namespace impl {
/**
 * @brief Converts <T, shape[0], ..., shape[n]> to type T[shape[0]]...[shape[n]]
 */
template <typename T, int... DataArrayShape> struct arrayify_datatype_expand;

/**
 * @brief Generates an array type T[...] from arguments for a Kokkos View.
 *
 * @tparam T base type (usually type_real)
 * @tparam ElementSubspaceShapeIntegerSequence point index shape
 * @tparam PointwiseShapeIntegerSequence tensor shape at each point
 */
template <typename T, typename ElementSubspaceShapeIntegerSequence,
          typename PointwiseShapeIntegerSequence>
struct arrayify_datatype {
public:
  using type = decltype(type_expand(ElementSubspaceShapeIntegerSequence(),
                                    PointwiseShapeIntegerSequence()))::type;

private:
  template <int... DataArrayShape>
  static arrayify_datatype_expand<T, DataArrayShape...>
  type_expand(std::integer_sequence<int, DataArrayShape...> integer_sequence) {
    return {};
  }
  template <int... SubspaceShape, int... PointwiseShape>
  static arrayify_datatype_expand<T, SubspaceShape..., PointwiseShape...>
  type_expand(std::integer_sequence<int, SubspaceShape...> integer_sequence1,
              std::integer_sequence<int, PointwiseShape...> integer_sequence2) {
    return {};
  }
};

template <typename T, int NextInt, int... DataArrayShape>
struct arrayify_datatype_expand<T, NextInt, DataArrayShape...> {
  using type = arrayify_datatype_expand<T[NextInt], DataArrayShape...>::type;
};
template <typename T> struct arrayify_datatype_expand<T> {
  using type = T;
};

// ==============================================

/**
 * @brief Converts the tuple <Dim,NGLL> to the integer sequence <NGLL,...,NGLL>
 * of size Dim. This is the sequence that should be passed in as
 * ElementSubspaceShapeIntegerSequence
 */
template <int SubspaceDimension, int NGLL> struct subspace_shape {
  using type =
      decltype(type_expand(subspace_shape<SubspaceDimension - 1, NGLL>()));

private:
  template <int... Vals>
  std::integer_sequence<int, Vals..., NGLL>
  type_expand(std::integer_sequence<int, Vals...> integer_sequence) {

  };
};
template <int NGLL> struct subspace_shape<0, NGLL> {
  using type = std::integer_sequence<int>;
};

} // namespace impl

// ==============================================

template <typename T, specfem::element::dimension_tag DimensionTag,
          int ChunkSize, int NGLL, int ChunkSubspaceDimension,
          typename PointwiseShapeIntegerSequence, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged>>
struct ChunkNDimViewType
    : public Kokkos::View<
          typename impl::arrayify_datatype<
              typename specfem::datatype::simd<T, UseSIMD>::datatype,
              impl::subspace_shape<ChunkSubspaceDimension, NGLL>,
              PointwiseShapeIntegerSequence>::type,
          MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type =
      Kokkos::View<typename impl::arrayify_datatype<
                       typename specfem::datatype::simd<T, UseSIMD>::datatype,
                       impl::subspace_shape<ChunkSubspaceDimension, NGLL>,
                       PointwiseShapeIntegerSequence>::type,
                   MemorySpace, MemoryTraits>; ///< Underlying data type used to
                                               ///< store values
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag
  using index_type =
      typename specfem::point::index<dimension_tag,
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
      specfem::datatype::AccessorType::chunk_element; ///< Accessor type for
                                                      ///< identifying the
                                                      ///< class

  constexpr static int chunk_size = ChunkSize; ///< Number of elements in
                                               ///< the chunk
  constexpr static int ngll = NGLL;            ///< Number of GLL points in
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
  ChunkNDimViewType() = default;

  /**
   * @brief Construct a new ChunkNDimViewType object within
   * ScratchMemorySpace.
   * Allocates an unmanaged view within ScratchMemorySpace. Useful for
   * generating scratch views.
   *
   * @tparam ScratchMemorySpace Memory space of the view
   * @param scratch_memory_space Memory space of the view
   */
  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION
  ChunkNDimViewType(const ScratchMemorySpace &scratch_memory_space)
      : Kokkos::View<typename impl::arrayify_datatype<
                         typename specfem::datatype::simd<T, UseSIMD>::datatype,
                         impl::subspace_shape<ChunkSubspaceDimension, NGLL>,
                         PointwiseShapeIntegerSequence>::type,
                     MemorySpace, MemoryTraits>(scratch_memory_space) {}
  ///@}

  using type::operator();

  /**
   * @brief Get scalar value by a point index.
   *
   * @param index Point index
   * @param icomp...
   */

  template <typename... ComponentTypes>
  KOKKOS_INLINE_FUNCTION constexpr value_type &
  operator()(index_type index, ComponentTypes... components) {
    if constexpr (dimension_tag == element::dimension_tag::dim3) {
      return (*this)(index.ispec, index.iz, index.iy, index.ix, components...);
    } else {
      return (*this)(index.ispec, index.iz, index.ix, components...);
    }
  }
};

} // namespace specfem::datatype
