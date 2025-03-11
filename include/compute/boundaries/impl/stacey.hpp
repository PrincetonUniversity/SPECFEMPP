#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "mesh/mesh.hpp"
#include "point/boundary.hpp"
#include "point/coordinates.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem {
namespace compute {
namespace impl {
namespace boundaries {

struct stacey {
private:
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::stacey; ///< Boundary tag
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension

public:
  using BoundaryTagView =
      Kokkos::View<specfem::element::boundary_tag_container ***,
                   Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  using EdgeNormalView = Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                      Kokkos::DefaultExecutionSpace>;
  using EdgeWeightView = Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                      Kokkos::DefaultExecutionSpace>;

  BoundaryTagView quadrature_point_boundary_tag; ///< Boundary tag for every
  ///< quadrature point within an
  ///< element with Stacey boundary

  BoundaryTagView::HostMirror h_quadrature_point_boundary_tag; ///< Host mirror
                                                               ///< of boundary
                                                               ///< types

  EdgeNormalView edge_normal; ///< Normal vector to the edge for every
                              ///< quadrature point within an element with
                              ///< Stacey boundary
  EdgeWeightView edge_weight; ///< Edge weight used to compute integrals on the
                              ///< edge for every quadrature point within an
                              ///< element with Stacey boundary. Evaluates to 0
                              ///< for points not on the edge

  EdgeNormalView::HostMirror h_edge_normal; ///< Host mirror of edge normal

  EdgeWeightView::HostMirror h_edge_weight; ///< Host mirror of edge weight

  stacey() = default;

  stacey(const int nspec, const int ngllz, const int ngllx,
         const specfem::mesh::absorbing_boundary<specfem::dimension::type::dim2>
             &stacey,
         const specfem::compute::mesh_to_compute_mapping &mapping,
         const specfem::compute::quadrature &quadrature,
         const specfem::compute::partial_derivatives &partial_derivatives,
         const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
         std::vector<specfem::element::boundary_tag_container> &boundary_tag);

  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const specfem::point::index<dimension> &index,
                 specfem::point::boundary<boundary_tag, dimension, false>
                     &boundary) const {

    boundary.tag +=
        quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);

    boundary.edge_normal(0) = edge_normal(index.ispec, index.iz, index.ix, 0);
    boundary.edge_normal(1) = edge_normal(index.ispec, index.iz, index.ix, 1);
    boundary.edge_weight = edge_weight(index.ispec, index.iz, index.ix);

    return;
  }

  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const specfem::point::index<dimension> &index,
                 specfem::point::boundary<
                     specfem::element::boundary_tag::composite_stacey_dirichlet,
                     dimension, false> &boundary) const {

    boundary.tag +=
        quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);

    boundary.edge_normal(0) = edge_normal(index.ispec, index.iz, index.ix, 0);
    boundary.edge_normal(1) = edge_normal(index.ispec, index.iz, index.ix, 1);
    boundary.edge_weight = edge_weight(index.ispec, index.iz, index.ix);

    return;
  }

  KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
      const specfem::point::simd_index<dimension> &index,
      specfem::point::boundary<boundary_tag, dimension, true> &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(index.ispec + lane,
                                                            index.iz, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.edge_normal(0))
        .copy_from(&edge_normal(index.ispec, index.iz, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_normal(1))
        .copy_from(&edge_normal(index.ispec, index.iz, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_weight)
        .copy_from(&edge_weight(index.ispec, index.iz, index.ix), tag_type());
  }

  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const specfem::point::simd_index<dimension> &index,
                 specfem::point::boundary<
                     specfem::element::boundary_tag::composite_stacey_dirichlet,
                     dimension, true> &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(index.ispec + lane,
                                                            index.iz, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.edge_normal(0))
        .copy_from(&edge_normal(index.ispec, index.iz, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_normal(1))
        .copy_from(&edge_normal(index.ispec, index.iz, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_weight)
        .copy_from(&edge_weight(index.ispec, index.iz, index.ix), tag_type());
  }

  inline void load_on_host(const specfem::point::index<dimension> &index,
                           specfem::point::boundary<boundary_tag, dimension,
                                                    false> &boundary) const {
    boundary.tag +=
        h_quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);

    boundary.edge_normal(0) = h_edge_normal(index.ispec, index.iz, index.ix, 0);
    boundary.edge_normal(1) = h_edge_normal(index.ispec, index.iz, index.ix, 1);

    boundary.edge_weight = h_edge_weight(index.ispec, index.iz, index.ix);

    return;
  }

  inline void
  load_on_host(const specfem::point::index<dimension> &index,
               specfem::point::boundary<
                   specfem::element::boundary_tag::composite_stacey_dirichlet,
                   dimension, false> &boundary) const {
    boundary.tag +=
        h_quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);

    boundary.edge_normal(0) = h_edge_normal(index.ispec, index.iz, index.ix, 0);
    boundary.edge_normal(1) = h_edge_normal(index.ispec, index.iz, index.ix, 1);

    boundary.edge_weight = h_edge_weight(index.ispec, index.iz, index.ix);

    return;
  }

  inline void load_on_host(
      const specfem::point::simd_index<dimension> &index,
      specfem::point::boundary<boundary_tag, dimension, true> &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.edge_normal(0))
        .copy_from(&h_edge_normal(index.ispec, index.iz, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_normal(1))
        .copy_from(&h_edge_normal(index.ispec, index.iz, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_weight)
        .copy_from(&h_edge_weight(index.ispec, index.iz, index.ix), tag_type());

    return;
  }

  inline void
  load_on_host(const specfem::point::simd_index<dimension> &index,
               specfem::point::boundary<
                   specfem::element::boundary_tag::composite_stacey_dirichlet,
                   dimension, true> &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(index.ispec + lane,
                                                            index.iz, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.edge_normal(0))
        .copy_from(&h_edge_normal(index.ispec, index.iz, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_normal(1))
        .copy_from(&h_edge_normal(index.ispec, index.iz, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_weight)
        .copy_from(&h_edge_weight(index.ispec, index.iz, index.ix), tag_type());

    return;
  }
};
} // namespace boundaries
} // namespace impl
} // namespace compute
} // namespace specfem
