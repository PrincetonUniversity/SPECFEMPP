#ifndef _POINT_BOUNDARY_HPP
#define _POINT_BOUNDARY_HPP

#include "datatypes/point_view.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/boundary.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {
template <specfem::element::boundary_tag BoundaryTag, bool UseSIMD>
struct boundary;

template <bool UseSIMD>
struct boundary<specfem::element::boundary_tag::none, UseSIMD> {
private:
  using value_type = typename specfem::datatype::simd_like<
      specfem::element::boundary_tag_container, type_real, UseSIMD>::datatype;

public:
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto boundary_tag = specfem::element::boundary_tag::none;
  constexpr static bool isPointBoundaryType = true;

  KOKKOS_FUNCTION
  boundary() = default;

  value_type tag;
};

template <bool UseSIMD>
struct boundary<specfem::element::boundary_tag::acoustic_free_surface, UseSIMD>
    : public boundary<specfem::element::boundary_tag::none, UseSIMD> {
public:
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::acoustic_free_surface;

  KOKKOS_FUNCTION
  boundary() = default;

  KOKKOS_FUNCTION
  boundary(const specfem::point::boundary<
           specfem::element::boundary_tag::composite_stacey_dirichlet, UseSIMD>
               &boundary);
};

template <bool UseSIMD>
struct boundary<specfem::element::boundary_tag::stacey, UseSIMD>
    : public boundary<specfem::element::boundary_tag::none, UseSIMD> {
private:
  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;

public:
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto boundary_tag = specfem::element::boundary_tag::stacey;

  KOKKOS_FUNCTION
  boundary() = default;

  KOKKOS_FUNCTION
  boundary(const specfem::point::boundary<
           specfem::element::boundary_tag::composite_stacey_dirichlet, UseSIMD>
               &boundary);

  datatype edge_weight = 0.0;
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD> edge_normal = {
    0.0, 0.0
  };
};

template <bool UseSIMD>
struct boundary<specfem::element::boundary_tag::composite_stacey_dirichlet,
                UseSIMD>
    : public boundary<specfem::element::boundary_tag::stacey, UseSIMD> {
public:
  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::composite_stacey_dirichlet;

  KOKKOS_FUNCTION
  boundary() = default;
};

template <bool UseSIMD>
KOKKOS_FUNCTION
specfem::point::boundary<specfem::element::boundary_tag::acoustic_free_surface,
                         UseSIMD>::
    boundary(const specfem::point::boundary<
             specfem::element::boundary_tag::composite_stacey_dirichlet,
             UseSIMD> &boundary) {
  this->tag = boundary.tag;
}

template <bool UseSIMD>
KOKKOS_FUNCTION
specfem::point::boundary<specfem::element::boundary_tag::stacey, UseSIMD>::
    boundary(const specfem::point::boundary<
             specfem::element::boundary_tag::composite_stacey_dirichlet,
             UseSIMD> &boundary) {
  this->tag = boundary.tag;
  this->edge_weight = boundary.edge_weight;
  this->edge_normal = boundary.edge_normal;
}

} // namespace point
} // namespace specfem
#endif
