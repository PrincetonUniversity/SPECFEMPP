#pragma once

#include "enumerations/specfem_enums.hpp"
#include "point/coordinates.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace iterator {

namespace impl {

template <specfem::dimension::type DimensionType> struct index_type;

template <> struct index_type<specfem::dimension::type::dim2> {

  constexpr static auto dimension = specfem::dimension::type::dim2;
  const int iedge;
  specfem::point::index<dimension> self_index;
  specfem::point::index<dimension> coupled_index;

  KOKKOS_INLINE_FUNCTION
  index_type(const int iedge,
             const specfem::point::index<dimension> &self_index,
             const specfem::point::index<dimension> &coupled_index)
      : iedge(iedge), self_index(self_index), coupled_index(coupled_index) {}
};
} // namespace impl

template <specfem::dimension::type DimensionType> struct edge;

template <> struct edge<specfem::dimension::type::dim2> {

  constexpr static auto dimension = specfem::dimension::type::dim2;
  int iedge;
  int self_element;
  int coupled_element;
  specfem::enums::edge::type self_edge;
  specfem::enums::edge::type coupled_edge;
  int npoints;

  KOKKOS_INLINE_FUNCTION
  edge(const int iedge, const int self_element, const int coupled_element,
       const specfem::enums::edge::type self_edge,
       const specfem::enums::edge::type coupled_edge, const int ngll)
      : iedge(iedge), self_element(self_element),
        coupled_element(coupled_element), self_edge(self_edge),
        coupled_edge(coupled_edge), npoints(ngll) {}

  KOKKOS_INLINE_FUNCTION
  int edge_size() const { return npoints; }

  KOKKOS_INLINE_FUNCTION
  impl::index_type<dimension> operator()(const int i) const {
    const auto self_index = this->self_index(i);
    const auto coupled_index = this->coupled_index(i);
    return impl::index_type<dimension>(iedge, self_index, coupled_index);
  }

private:
  KOKKOS_INLINE_FUNCTION
  specfem::point::index<dimension> self_index(const int ipoint) const {
    switch (self_edge) {
    case specfem::enums::edge::type::BOTTOM:
      return { self_element, 0, ipoint };
      break;
    case specfem::enums::edge::type::TOP:
      return { self_element, npoints - 1, npoints - 1 - ipoint };
      break;
    case specfem::enums::edge::type::LEFT:
      return { self_element, ipoint, 0 };
      break;
    case specfem::enums::edge::type::RIGHT:
      return { self_element, npoints - 1 - ipoint, npoints - 1 };
      break;
    default:
      DEVICE_ASSERT(false, "Invalid edge type");
    }
  }

  KOKKOS_INLINE_FUNCTION
  specfem::point::index<dimension> coupled_index(const int ipoint) const {
    switch (coupled_edge) {
    case specfem::enums::edge::type::BOTTOM:
      return { coupled_element, 0, npoints - 1 - ipoint };
      break;
    case specfem::enums::edge::type::TOP:
      return { coupled_element, npoints - 1, ipoint };
      break;
    case specfem::enums::edge::type::LEFT:
      return { coupled_element, npoints - 1 - ipoint, npoints - 1 };
      break;
    case specfem::enums::edge::type::RIGHT:
      return { coupled_element, ipoint, 0 };
      break;
    default:
      DEVICE_ASSERT(false, "Invalid edge type");
    }
  }
};

} // namespace iterator

namespace policy {

template <specfem::dimension::type DimensionType, typename... PolicyTraits>
struct element_edge : public Kokkos::TeamPolicy<PolicyTraits...> {
public:
  constexpr static bool isChunkPolicy = false;
  constexpr static bool isRangePolicy = false;
  constexpr static auto dimension = DimensionType;

  using PolicyType = Kokkos::TeamPolicy<PolicyTraits...>;

  using member_type = typename PolicyType::member_type;

private:
  using IndexViewType =
      Kokkos::View<int *, typename member_type::execution_space::memory_space>;
  using EdgeViewType =
      Kokkos::View<specfem::enums::edge::type *,
                   typename member_type::execution_space::memory_space>;

public:
  element_edge(const IndexViewType _self_indices,
               const IndexViewType _coupled_indices,
               const EdgeViewType _self_edges,
               const EdgeViewType _coupled_edges, const int npoints)
      : PolicyType(_self_indices.extent(0), Kokkos::AUTO, Kokkos::AUTO),
        self_indices(_self_indices), coupled_indices(_coupled_indices),
        self_edges(_self_edges), coupled_edges(_coupled_edges),
        npoints(npoints) {

    const int nedges = _self_indices.extent(0);

    if (nedges != _coupled_indices.extent(0)) {
      throw std::runtime_error("Number of edges must be equal");
    }

    if (nedges != _self_edges.extent(0)) {
      throw std::runtime_error("Number of edges must be equal");
    }

    if (nedges != _coupled_edges.extent(0)) {
      throw std::runtime_error("Number of edges must be equal");
    }
  }

  inline PolicyType get_policy() const { return *this; }

  KOKKOS_INLINE_FUNCTION
  auto league_iterator(const int iedge) const {
    return specfem::iterator::edge<dimension>(
        iedge, self_indices(iedge), coupled_indices(iedge), self_edges(iedge),
        coupled_edges(iedge), npoints);
  }

private:
  IndexViewType self_indices;
  IndexViewType coupled_indices;
  EdgeViewType self_edges;
  EdgeViewType coupled_edges;
  int npoints;
};

} // namespace policy
} // namespace specfem
