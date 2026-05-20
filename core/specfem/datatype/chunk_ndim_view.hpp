#pragma once

#include "accessor_type.hpp"
#include "simd.hpp"
#include "specfem/element/tags.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <utility>

// Forward declarations
namespace specfem::point {
template <specfem::element::dimension_tag DimensionTag, bool UseSIMD>
struct index;
template <specfem::element::dimension_tag DimensionTag> struct face_index;
template <specfem::element::dimension_tag DimensionTag> struct edge_index;
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
template <typename T, int chunk_size,
          typename ElementSubspaceShapeIntegerSequence,
          typename PointwiseShapeIntegerSequence>
struct arrayify_datatype {
private:
  template <int... SubspaceShape, int... PointwiseShape>
  static arrayify_datatype_expand<T, chunk_size, SubspaceShape...,
                                  PointwiseShape...>
  type_expand(std::integer_sequence<int, SubspaceShape...> integer_sequence1,
              std::integer_sequence<int, PointwiseShape...> integer_sequence2) {
    return {};
  }

public:
  using type = decltype(type_expand(ElementSubspaceShapeIntegerSequence(),
                                    PointwiseShapeIntegerSequence()))::type;
};

template <typename T, int NextInt, int... DataArrayShape>
struct arrayify_datatype_expand<T, NextInt, DataArrayShape...> {
  using type = arrayify_datatype_expand<T, DataArrayShape...>::type[NextInt];
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
private:
  template <int... Vals>
  static std::integer_sequence<int, Vals..., NGLL>
  type_expand(std::integer_sequence<int, Vals...> integer_sequence) {};

public:
  using type = decltype(type_expand(
      typename subspace_shape<SubspaceDimension - 1, NGLL>::type()));
};
template <int NGLL> struct subspace_shape<0, NGLL> {
  using type = std::integer_sequence<int>;
};

// mini-test: 8-7 tensor on NGLL=5 2D element (chunk size = 1)
static_assert(
    std::is_same<arrayify_datatype<type_real, 1,
                                   typename impl::subspace_shape<2, 5>::type,
                                   std::integer_sequence<int, 8, 7>>::type,
                 type_real[1][5][5][8][7]>());

} // namespace impl

// ==============================================

template <typename T, specfem::element::dimension_tag DimensionTag,
          int ChunkSize, int NGLL, int ChunkSubspaceDimension,
          typename PointwiseShapeIntegerSequence,
          specfem::datatype::AccessorType AccessorType, bool UseSIMD = false,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged>>
struct ChunkNDimViewType
    : public Kokkos::View<
          typename impl::arrayify_datatype<
              typename specfem::datatype::simd<T, UseSIMD>::datatype, ChunkSize,
              typename impl::subspace_shape<ChunkSubspaceDimension, NGLL>::type,
              PointwiseShapeIntegerSequence>::type,
          MemorySpace, MemoryTraits> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using type = Kokkos::View<
      typename impl::arrayify_datatype<
          typename specfem::datatype::simd<T, UseSIMD>::datatype, ChunkSize,
          typename impl::subspace_shape<ChunkSubspaceDimension, NGLL>::type,
          PointwiseShapeIntegerSequence>::type,
      MemorySpace, MemoryTraits>; ///< Underlying data type used to
                                  ///< store values
  using value_type = typename type::value_type; ///< Value type used to store
                                                ///< the elements of the array
  using base_type = T;                          ///< Base type of the array
  using index_type = std::conditional_t<
      AccessorType == specfem::datatype::AccessorType::chunk_element,
      specfem::point::index<DimensionTag, UseSIMD>,
      std::conditional_t<
          AccessorType == specfem::datatype::AccessorType::chunk_face,
          specfem::point::face_index<DimensionTag>,
          std::conditional_t<
              AccessorType == specfem::datatype::AccessorType::chunk_face,
              specfem::point::edge_index<DimensionTag>, void>>>; ///< index type
                                                                 ///< for
                                                                 ///< accessing
                                                                 ///< at GLL
                                                                 ///< level
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag

  constexpr static auto accessor_type = AccessorType; ///< Accessor type for
                                                      ///< identifying the
                                                      ///< class
  constexpr static int chunk_size = ChunkSize;        ///< Number of elements in
                                                      ///< the chunk
  constexpr static int ngll = NGLL;           ///< Number of GLL points in
                                              ///< each element
  constexpr static bool using_simd = UseSIMD; ///< Use SIMD datatypes for the
                                              ///< array. If false,
                                              ///< std::is_same<value_type,
                                              ///< base_type>::value is true
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
      : type(scratch_memory_space) {}
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
    if constexpr (accessor_type ==
                  specfem::datatype::AccessorType::chunk_element) {
      if constexpr (dimension_tag == element::dimension_tag::dim3) {
        return (*this)(index.ispec, index.iz, index.iy, index.ix,
                       components...);
      } else {
        return (*this)(index.ispec, index.iz, index.ix, components...);
      }
    } else if constexpr (accessor_type ==
                         specfem::datatype::AccessorType::chunk_face) {
      return (*this)(index.ispec, index.ipoint_i, index.ipoint_j);
    } else if constexpr (accessor_type ==
                         specfem::datatype::AccessorType::chunk_edge) {
      return (*this)(index.ispec, index.ipoint);
    }
  }
};

} // namespace specfem::datatype
