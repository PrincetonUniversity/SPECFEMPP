#pragma once

#include "parallel_configuration/chunk_config.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <mdspan/mdspan.hpp>
#include <tuple>

namespace specfem {
namespace kokkos {

namespace impl {
template <int ElementChunkSize, typename Extents>
constexpr std::size_t chunk_size(const Extents &extents) {
  std::size_t size = ElementChunkSize;
  for (std::size_t i = 1; i < Extents::rank(); ++i) {
    size *= extents.extent(i);
  }
  return size;
}

template <int ElementChunkSize, typename Extents>
constexpr std::size_t number_of_tiles(const Extents &extents, std::size_t dim) {
  if (dim == 0)
    return ((extents.extent(0) / ElementChunkSize) +
            (extents.extent(0) % ElementChunkSize != 0 ? 1 : 0));

  return 1;
}

template <int ElementChunkSize, typename Extents>
constexpr std::size_t fwd_prod_of_tiles(const Extents &extents,
                                        const std::size_t idim) {
  std::size_t prod = 1;
  for (std::size_t i = 0; i < idim; ++i) {
    prod *= impl::number_of_tiles<ElementChunkSize>(extents, i);
  }

  return prod;
}

template <int ElementChunkSize, typename Extents>
constexpr std::size_t tile_size(const Extents &extents,
                                const std::size_t &dim) {
  if (dim == 0)
    return ElementChunkSize;

  return extents.extent(dim);
}

template <int ElementChunkSize, typename Extents>
constexpr std::size_t fwd_prod_of_tile_size(const Extents &extents,
                                            const std::size_t idim) {
  std::size_t prod = 1;
  for (std::size_t i = 0; i < idim; ++i) {
    prod *= tile_size<ElementChunkSize>(extents, i);
  }
  return prod;
}
} // namespace impl

template <int ElementChunkSize> struct DomainViewMapping2D {
public:
  template <class Extents> struct mapping {
  public:
    using extents_type = Extents;
    using rank_type = typename extents_type::rank_type;
    using size_type = typename extents_type::size_type;
    using layout_type = mapping;

    mapping() noexcept = default;
    mapping &operator=(const mapping &) noexcept = default;

    mapping(const mapping &other) noexcept
        : extents_(other.extents_),
          chunk_size(impl::chunk_size<ElementChunkSize>(other.extents_)) {}

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
    constexpr size_type operator()(IndexType... indices) const noexcept {
      static_assert(sizeof...(indices) == extents_type::rank(),
                    "Number of indices must match the rank of extents.");
      return tile_offset(indices...) + offset_in_tile(indices...);
    }

    // Mapping is always unique
    static constexpr bool is_always_unique() noexcept { return true; }

    constexpr bool is_exhaustive() noexcept {
      // Only exhaustive if tiles fit exactly into extents
      for (std::size_t i = 0; i < extents_type::rank(); ++i) {
        const int tile_size = impl::tile_size<ElementChunkSize>(extents_, i);
        if ((extents_.extent(i) % tile_size) != 0) {
          return false;
        }
      }
      return true;
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
    constexpr size_type tile_offset(std::size_t ispec,
                                    IndexType... /* indices */) const noexcept {

      // size_type index_array[] = { static_cast<size_type>(indices)... };

      // size_type t_index[extents_type::rank()];

      // for (std::size_t i = 0; i < extents_type::rank(); ++i) {
      //   t_index[i] =
      //       index_array[i] / impl::tile_size<ElementChunkSize>(extents_, i);
      // }

      // size_type offset = t_index[0];

      // // Hardcoded tiles with Layout left
      // for (std::size_t i = 1; i < extents_type::rank(); ++i) {
      //   offset +=
      //       t_index[i] * impl::fwd_prod_of_tiles<ElementChunkSize>(extents_,
      //       i);
      // }

      return (ispec / ElementChunkSize) * chunk_size;
    } // tile_offset

    template <typename... IndexType>
    constexpr size_type offset_in_tile(IndexType... indices) const noexcept {
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

  View(const View &other) : base_type(other), _mapping(other._mapping) {}

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

  base_type get_base_view() const {
    return static_cast<const base_type &>(*this);
  }

  mapping_type get_mapping() const { return _mapping; }

private:
  mapping_type _mapping;
};

template <std::size_t Rank>
using chunked_tiled_layout2d = Kokkos::dextents<std::size_t, Rank>;

template <typename T, std::size_t Rank, typename MemorySpace>
using DomainView2d =
    View<T, chunked_tiled_layout2d<Rank>,
         DomainViewMapping2D<specfem::parallel_config::storage_chunk_size>,
         MemorySpace>;

template <typename ViewType>
specfem::kokkos::View<typename ViewType::value_type,
                      typename ViewType::extents_type,
                      typename ViewType::layout_type, Kokkos::HostSpace>
create_mirror_view(const ViewType &view) {
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
void deep_copy(const DstViewType &dst, const SrcViewType &src) {
  Kokkos::deep_copy(dst.get_base_view(), src.get_base_view());
}

// template <int TileSize> struct DomainViewMapping2D {
// public:
//   template <class Extents> struct mapping {
//   public:
//     using extents_type = Extents;
//     using rank_type = typename extents_type::rank_type;
//     using size_type = typename extents_type::size_type;
//     using layout_type = mapping;
//     constexpr static int tile_size = TileSize;

//     mapping() noexcept = default;
//     mapping &operator=(const mapping &) noexcept = default;

//     mapping(const mapping &other) noexcept : extents_(other.extents_) {}

//     mapping(const extents_type &extents) : extents_(extents) {}

//     MDSPAN_TEMPLATE_REQUIRES(
//         class OtherExtents,
//         /* requires */ (
//             ::std::is_constructible<extents_type, OtherExtents>::value))
//     MDSPAN_CONDITIONAL_EXPLICIT(
//         (!::std::is_convertible<OtherExtents, extents_type>::value))
//     constexpr mapping(const mapping<OtherExtents> &input_mapping) noexcept
//         : extents_(input_mapping.extents()) {}

//     //------------------------------------------------------------
//     // Required members

//     constexpr const extents_type &extents() const { return extents_; }

//     constexpr size_type required_span_size() const {
//       return this->number_of_tiles() * this->chunk_size();
//     }

//     template <typename... IndexType>
//     constexpr size_type operator()(IndexType... indices) const {
//       return tile_offset(indices...) + offset_in_tile(indices...);
//     }

//     // Mapping is always unique
//     static constexpr bool is_always_unique() noexcept { return true; }
//     // Only exhaustive if extents_.extent(0) % column_tile_size == 0, so not
//     // always
//     static constexpr bool is_always_exhaustive() noexcept { return false; }
//     // There is not always a regular stride between elements in a given
//     // dimension
//     static constexpr bool is_always_strided() noexcept { return false; }

//     static constexpr bool is_unique() noexcept { return true; }

//     constexpr bool is_exhaustive() const noexcept {
//       // Only exhaustive if extents fit exactly into tile sizes...
//       return (extents_.extent(0) % tile_size == 0);
//     }
//     // There are some circumstances where this is strided, but we're not
//     // concerned about that optimization, so we're allowed to just return
//     false
//     // here
//     constexpr bool is_strided() const noexcept { return tile_size == 1; }

//   private:
//     extents_type extents_; ///< Extents of the view

//     constexpr size_type number_of_tiles() const {
//       return extents_.extent(0) / tile_size +
//              size_type(extents_.extent(0) % tile_size != 0);
//     } // tile_size

//     constexpr size_type chunk_size() const {
//       return tile_size * impl::compute_size(extents_);
//     } // tile_size

//     template <typename... IndexType>
//     constexpr size_type tile_offset(const size_type ispec, const IndexType...
//     /*indices*/) const {
//       return this->chunk_size() * (ispec / tile_size);
//     } // tile_offset

//     constexpr size_type offset_in_tile(const size_type ispec,
//                                        const size_type iz,
//                                        const size_type ix) const {

//       auto t_ispec = ispec % tile_size;

//       return t_ispec * extents_.extent(1) * extents_.extent(2) +
//              iz * extents_.extent(2) + ix;
//     } // offset_in_tile

//     constexpr size_type offset_in_tile(const size_type ispec,
//                                        const size_type iz, const size_type
//                                        ix, const size_type N) const {

//       auto t_ispec = ispec % tile_size;

//       return t_ispec * extents_.extent(1) * extents_.extent(2) *
//                  extents_.extent(3) +
//              iz * extents_.extent(2) * extents_.extent(3) +
//              ix * extents_.extent(3) + N;
//     } // offset_in_tile
//   };

// }; // struct DomainViewMapping

// template <typename T, typename Extents, int TileSize, typename MemorySpace>
// class domain_view2D
//     : public Kokkos::mdspan<T, Extents, DomainViewMapping2D<TileSize> >,
//       public Kokkos::View<T *, MemorySpace> {

// public:
//   using base_type = Kokkos::View<T *, MemorySpace>;
//   using view_type = Kokkos::mdspan<T, Extents, DomainViewMapping2D<TileSize>
//   >; using mapping_type = typename view_type::mapping_type; constexpr static
//   int tile_size = mapping_type::tile_size; using memory_space = MemorySpace;
//   using value_type = T;

// private:
//   const base_type get_base_view() const {
//     return static_cast<base_type>(*this);
//   }

//   constexpr mapping_type mapping() const { return view_type::mapping(); }

// public:
//   domain_view2D() = default;

//   domain_view2D(const std::string &name, const mapping_type &mapping)
//       : base_type(name, mapping.required_span_size()),
//         view_type(base_type::data(), mapping) {}

//   domain_view2D(const domain_view2D &other)
//       : base_type(other.get_base_view()),
//         view_type(other.data(), other.mapping()) {}

//   domain_view2D(const std::string &name, const int nspec, const int ngllz,
//                 const int ngllx)
//       : base_type(name, nspec * ngllz * ngllx),
//         view_type(base_type::data(), Extents{ nspec, ngllz, ngllx }) {}

//   domain_view2D(const std::string &name, const int nspec, const int ngllz,
//                 const int ngllx, const int N)
//       : base_type(name, nspec * ngllz * ngllx * N),
//         view_type(base_type::data(),
//                   Extents{ nspec, ngllz, ngllx, N }) {}

//   domain_view2D(const base_type &view, const mapping_type &mapping)
//       : base_type(view), view_type(view.data(), mapping) {}

//   constexpr static size_t rank() { return view_type::rank(); }

//   constexpr size_t extent(const size_t i) const { return
//   view_type::extent(i); }

//   template <typename... IndexType>
//   KOKKOS_FORCEINLINE_FUNCTION T &operator()(IndexType... indices) const {
//     return view_type::operator()(indices...);
//   }

//   template <typename... IndexType>
//   KOKKOS_FORCEINLINE_FUNCTION T &operator()(IndexType... indices) {
//     return view_type::operator()(indices...);
//   }

//   using HostMirror = domain_view2D<T, Extents, tile_size, Kokkos::HostSpace>;

//   template <typename DstView, typename SrcView>
//   friend void deep_copy(DstView &dst, const SrcView &src);

//   template <typename DomainViewType>
//   friend specfem::kokkos::domain_view2D<typename DomainViewType::value_type,
//                                         typename
//                                         DomainViewType::extents_type,
//                                         DomainViewType::tile_size,
//                                         Kokkos::HostSpace>
//   create_mirror_view(const DomainViewType view);

// }; // class domain_view2D

// template <typename DstView, typename SrcView>
// void deep_copy(DstView &dst, const SrcView &src) {
//   Kokkos::deep_copy(dst.get_base_view(), src.get_base_view());
// }

// template <typename DomainViewType>
// specfem::kokkos::domain_view2D<typename DomainViewType::value_type,
//                                typename DomainViewType::extents_type,
//                                DomainViewType::tile_size, Kokkos::HostSpace>
// create_mirror_view(const DomainViewType view) {
//   if constexpr (std::is_same_v<typename DomainViewType::memory_space,
//                                Kokkos::HostSpace>) {
//     return specfem::kokkos::domain_view2D<typename
//     DomainViewType::value_type,
//                                           typename
//                                           DomainViewType::extents_type,
//                                           DomainViewType::tile_size,
//                                           Kokkos::HostSpace>(
//         view.get_base_view(), view.mapping());
//   } else {
//     return specfem::kokkos::domain_view2D<typename
//     DomainViewType::value_type,
//                                           typename
//                                           DomainViewType::extents_type,
//                                           DomainViewType::tile_size,
//                                           Kokkos::HostSpace>("mirror",
//                                                              view.mapping());
//   }
// } // create_mirror_view

} // namespace kokkos
} // namespace specfem
