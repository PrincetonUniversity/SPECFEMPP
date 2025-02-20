#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
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

struct acoustic_free_surface {
private:
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::acoustic_free_surface; ///< Boundary tag
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension

public:
  using BoundaryTagView =
      Kokkos::View<specfem::element::boundary_tag_container ***,
                   Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  BoundaryTagView quadrature_point_boundary_tag; ///< Boundary tag for every
                                                 ///< quadrature point within an
                                                 ///< element with acoustic free
                                                 ///< surface boundary

  BoundaryTagView::HostMirror h_quadrature_point_boundary_tag; ///< Host mirror
                                                               ///< of boundary
                                                               ///< types

  acoustic_free_surface() = default;

  acoustic_free_surface(
      const int nspec, const int ngllz, const int ngllx,
      const specfem::mesh::acoustic_free_surface<specfem::dimension::type::dim2>
          &acoustic_free_surface,
      const specfem::compute::mesh_to_compute_mapping &mapping,
      const specfem::compute::properties &properties,
      const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
      std::vector<specfem::element::boundary_tag_container> &boundary_tag);

  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const specfem::point::index<dimension> &index,
                 specfem::point::boundary<boundary_tag, dimension, false>
                     &boundary) const {

    boundary.tag +=
        quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);
    return;
  }

  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const specfem::point::index<dimension> &index,
                 specfem::point::boundary<
                     specfem::element::boundary_tag::composite_stacey_dirichlet,
                     dimension, false> &boundary) const {

    boundary.tag +=
        quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);
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

    return;
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

    return;
  }

  inline void load_on_host(const specfem::point::index<dimension> &index,
                           specfem::point ::boundary<boundary_tag, dimension,
                                                     false> &boundary) const {
    boundary.tag +=
        h_quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);
    return;
  }

  inline void
  load_on_host(const specfem::point::index<dimension> &index,
               specfem::point::boundary<
                   specfem::element::boundary_tag::composite_stacey_dirichlet,
                   dimension, false> &boundary) const {

    boundary.tag +=
        h_quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);
    return;
  }

  inline void load_on_host(
      const specfem::point::simd_index<dimension> &index,
      specfem::point::boundary<boundary_tag, dimension, true> &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.ix);
      }
    }
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

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.ix);
      }
    }
    return;
  }
};

} // namespace boundaries
} // namespace impl
} // namespace compute
} // namespace specfem
