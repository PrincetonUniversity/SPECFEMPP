#pragma once

#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::datatype::impl {

template <typename ViewType, typename PointViewType, typename IndexType>
struct VectorChunkSubview2D {
  ViewType &view;
  const IndexType &index;

  VectorChunkSubview2D(ViewType &view, const IndexType &index)
      : view(view), index(index) {}

  KOKKOS_INLINE_FUNCTION
  constexpr auto &operator()(const int &icomp) {
    return view(index.ispec, index.iz, index.ix, icomp);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr auto operator()(const int &icomp) const {
    return view(index.ispec, index.iz, index.ix, icomp);
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator=(const PointViewType &other) {
    for (int icomp = 0; icomp < PointViewType::components; ++icomp) {
      (*this)(icomp) = other(icomp);
    }
    return *this;
  }
};

template <typename ViewType, typename PointViewType, typename IndexType>
struct TensorChunkSubview2D {
  ViewType &view;
  const IndexType &index;

  KOKKOS_INLINE_FUNCTION
  constexpr static std::size_t rank() { return 2; }

  KOKKOS_INLINE_FUNCTION
  constexpr std::size_t extent(const size_t &i) const {
    return view.extent(i + 3);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr static std::size_t static_extent(const size_t &i) {
    return ViewType::static_extent(i + 3);
  }

  TensorChunkSubview2D(ViewType &view, const IndexType &index)
      : view(view), index(index) {}

  KOKKOS_INLINE_FUNCTION
  constexpr auto &operator()(const int &icomp, const int &idim) {
    return view(index.ispec, index.iz, index.ix, icomp, idim);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr auto operator()(const int &icomp, const int &idim) const {
    return view(index.ispec, index.iz, index.ix, icomp, idim);
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator=(const PointViewType &other) {
    for (int icomp = 0; icomp < PointViewType::components; ++icomp) {
      for (int idim = 0; idim < PointViewType::dimensions; ++idim) {
        (*this)(icomp, idim) = other(icomp, idim);
      }
    }
    return *this;
  }
};

} // namespace specfem::datatype::impl
