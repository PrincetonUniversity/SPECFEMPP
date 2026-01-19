#pragma once

#include "specfem/parallel_configuration.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <mdspan/mdspan.hpp>
#include <tuple>

namespace specfem {
namespace kokkos {

namespace impl {
template <int ElementChunkSize, typename Extents>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
chunk_size(const Extents &extents) {
  std::size_t size = ElementChunkSize;
  for (std::size_t i = 1; i < Extents::rank(); ++i) {
    size *= extents.extent(i);
  }
  return size;
}

template <int ElementChunkSize, typename Extents>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
number_of_tiles(const Extents &extents, std::size_t dim) {
  if (dim == 0)
    return ((extents.extent(0) / ElementChunkSize) +
            (extents.extent(0) % ElementChunkSize != 0 ? 1 : 0));

  return 1;
}

template <int ElementChunkSize, typename Extents>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
fwd_prod_of_tiles(const Extents &extents, const std::size_t idim) {
  std::size_t prod = 1;
  for (std::size_t i = 0; i < idim; ++i) {
    prod *= impl::number_of_tiles<ElementChunkSize>(extents, i);
  }

  return prod;
}

template <int ElementChunkSize, typename Extents>
KOKKOS_INLINE_FUNCTION constexpr std::size_t tile_size(const Extents &extents,
                                                       const std::size_t &dim) {
  if (dim == 0)
    return ElementChunkSize;

  return extents.extent(dim);
}

template <int ElementChunkSize, typename Extents>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
fwd_prod_of_tile_size(const Extents &extents, const std::size_t idim) {
  std::size_t prod = 1;
  for (std::size_t i = 0; i < idim; ++i) {
    prod *= tile_size<ElementChunkSize>(extents, i);
  }
  return prod;
}
} // namespace impl

template <specfem::dimension::type DimensionTag, int ElementChunkSize>
struct DomainViewMapping {
public:
  template <class Extents> struct mapping {
  public:
    using extents_type = Extents;
    using rank_type = typename extents_type::rank_type;
    using size_type = typename extents_type::size_type;
    using layout_type = mapping;

    KOKKOS_INLINE_FUNCTION
    mapping() noexcept = default;

    KOKKOS_INLINE_FUNCTION
    mapping &operator=(const mapping &) noexcept = default;

    KOKKOS_INLINE_FUNCTION
    mapping(const mapping &other) noexcept
        : extents_(other.extents_),
          chunk_size(impl::chunk_size<ElementChunkSize>(other.extents_)) {}

    KOKKOS_INLINE_FUNCTION
    mapping(const extents_type &extents) noexcept
        : extents_(extents),
          chunk_size(impl::chunk_size<ElementChunkSize>(extents)) {}

    MDSPAN_TEMPLATE_REQUIRES(
        class OtherExtents,
        /* requires */ (
            ::std::is_constructible<extents_type, OtherExtents>::value))
    MDSPAN_CONDITIONAL_EXPLICIT(
        (!::std::is_convertible<OtherExtents, extents_type>::value))
    constexpr mapping(const mapping<OtherExtents> &input_mapping) noexcept
        : extents_(input_mapping.extents()) {}
    //------------------------------------------------------------
    // Required members

    constexpr const extents_type &extents() const { return extents_; }

    constexpr size_type required_span_size() const noexcept {
      size_type total_size = chunk_size;
      for (std::size_t i = 0; i < extents_type::rank(); ++i) {
        total_size *= impl::number_of_tiles<ElementChunkSize>(extents_, i);
      }
      return total_size;
    }

    template <typename... IndexType>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    operator()(IndexType... indices) const noexcept {
      static_assert(sizeof...(indices) == extents_type::rank(),
                    "Number of indices must match the rank of extents.");

      return tile_offset(indices...) + offset_in_tile(indices...);
    }

    // Mapping is always unique
    static constexpr bool is_always_unique() noexcept { return true; }

    constexpr bool is_exhaustive() noexcept {
      // Only exhaustive if tiles fit exactly into extents
      return false;
    }

    static constexpr bool is_always_exhaustive() noexcept { return false; }

    constexpr bool is_strided() noexcept { return false; }

    static constexpr bool is_unique() noexcept { return true; }

    static constexpr bool is_always_strided() noexcept {
      // This is a simplification, as we can't guarantee stridedness in all
      // cases
      return false;
    }

  private:
    extents_type extents_; ///< Extents of the view
    size_type chunk_size;

    template <typename... IndexType>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    tile_offset(std::size_t ispec, IndexType... /* indices */) const noexcept {

      return (ispec / ElementChunkSize) * chunk_size;
    } // tile_offset

    template <typename... IndexType>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    offset_in_tile(IndexType... indices) const noexcept {
      static_assert(sizeof...(indices) == extents_type::rank(),
                    "Number of indices must match the rank of extents.");

      size_type index_array[] = { static_cast<size_type>(indices)... };

      std::size_t offset = index_array[0] % ElementChunkSize;

      // Hardcoded Layout left within tiles
      for (std::size_t i = 1; i < extents_type::rank(); ++i) {
        offset += index_array[i] *
                  impl::fwd_prod_of_tile_size<ElementChunkSize>(extents_, i);
      }

      return offset;
    } // offset_in_tile
  };
};

template <typename T, typename Extents, typename Layout, typename MemorySpace>
class View : public Kokkos::View<T *, MemorySpace> {

public:
  using value_type = T;
  using extents_type = Extents;
  using layout_type = Layout;
  using memory_space = MemorySpace;
  using mapping_type = typename Layout::template mapping<Extents>;
  using base_type = Kokkos::View<T *, MemorySpace>;

  template <typename mapping_type>
  View(const std::string &label, const mapping_type mapping)
      : _mapping(mapping), base_type(label, mapping.required_span_size()) {}

  template <typename... IndexType>
  View(const std::string &label, const IndexType &...indices)
      : View(label, mapping_type(Extents{ indices... })) {}

  KOKKOS_INLINE_FUNCTION
  View(const View &other) : base_type(other), _mapping(other._mapping) {}

  KOKKOS_INLINE_FUNCTION
  View() = default;

  using HostMirror =
      View<T, Extents, Layout, typename Kokkos::HostSpace::memory_space>;

public:
  template <typename... IndexType,
            std::enable_if_t<sizeof...(IndexType) == Extents::rank(), int> = 0>
  KOKKOS_INLINE_FUNCTION T &operator()(IndexType... indices) const {
    return static_cast<const base_type &>(*this)(_mapping(indices...));
  }

  template <typename... IndexType,
            std::enable_if_t<sizeof...(IndexType) == Extents::rank(), int> = 0>
  KOKKOS_INLINE_FUNCTION T &operator()(IndexType... indices) {
    return static_cast<base_type &>(*this)(_mapping(indices...));
  }

  KOKKOS_INLINE_FUNCTION T &operator[](const std::size_t index) const {
    return static_cast<const base_type &>(*this)[index];
  }

  KOKKOS_INLINE_FUNCTION T &operator[](const std::size_t index) {
    return static_cast<base_type &>(*this)[index];
  }

  KOKKOS_INLINE_FUNCTION
  base_type get_base_view() const {
    return static_cast<const base_type &>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  const mapping_type &get_mapping() const { return _mapping; }

private:
  mapping_type _mapping;
};

template <std::size_t Rank>
using chunked_tiled_layout2d = Kokkos::dextents<std::size_t, Rank>;

template <typename T, std::size_t Rank, typename MemorySpace>
using DomainView2d =
    View<T, chunked_tiled_layout2d<Rank>,
         DomainViewMapping<specfem::dimension::type::dim2,
                           specfem::parallel_configuration::storage_chunk_size>,
         MemorySpace>;

template <specfem::dimension::type DimensionTag, typename T, std::size_t Rank,
          typename MemorySpace>
using DomainView =
    View<T, Kokkos::dextents<std::size_t, Rank>,
         DomainViewMapping<DimensionTag,
                           specfem::parallel_configuration::storage_chunk_size>,
         MemorySpace>;

template <typename ViewType>
specfem::kokkos::View<typename ViewType::value_type,
                      typename ViewType::extents_type,
                      typename ViewType::layout_type, Kokkos::HostSpace>
create_mirror_view(const ViewType view) {
  if constexpr (std::is_same_v<typename ViewType::memory_space,
                               Kokkos::HostSpace>) {
    return view;
  } else if constexpr (std::is_same_v<
                           typename ViewType::memory_space,
                           Kokkos::DefaultExecutionSpace::memory_space>) {
    return specfem::kokkos::View<
        typename ViewType::value_type, typename ViewType::extents_type,
        typename ViewType::layout_type, Kokkos::HostSpace>("mirror",
                                                           view.get_mapping());
  } else {
    Kokkos::abort("Unsupported memory space for create_mirror_view");
  }
}

template <typename SrcViewType, typename DstViewType>
void deep_copy(const DstViewType dst, const SrcViewType src) {
  Kokkos::deep_copy(dst.get_base_view(), src.get_base_view());
}
} // namespace kokkos
} // namespace specfem
