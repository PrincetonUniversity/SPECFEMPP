#pragma once

#include "specfem/assembly/nonconforming_interfaces.hpp"
#include "specfem/element/tags.hpp"

namespace specfem::test_builder {

template <specfem::element::dimension_tag DimensionTag>
class NonconformingInterfacesPatch;

/**
 * @brief Patches assembly::nonconforming_interfaces to not require mesh and
 * edge types.
 *
 * nonconforming_interfaces only provides const access to the impl containers.
 * This class grants access to resizing these containers directly. Note that
 * const access still allows modification of values inside the views.
 */
template <>
class NonconformingInterfacesPatch<specfem::element::dimension_tag::dim2>
    : public specfem::assembly::nonconforming_interfaces<
          specfem::element::dimension_tag::dim2> {
  int ngllz;
  int ngllx;
  int nquad_intersection;

public:
  NonconformingInterfacesPatch(const int &ngllz, const int &ngllx,
                               const int &nquad_intersection)
      : ngllz(ngllz), ngllx(ngllx), nquad_intersection(nquad_intersection) {};

  template <specfem::element_coupling::interface_tag InterfaceTag,
            specfem::element::boundary_tag BoundaryTag,
            specfem::element_connections::type ConnectionTag>
  void reinit_container(const int &num_edges) {

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), CONNECTION_TAG(NONCONFORMING),
         INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
         BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                      COMPOSITE_STACEY_DIRICHLET)),
        CAPTURE(interface_container) {
          if constexpr (_interface_tag_ == InterfaceTag &&
                        _boundary_tag_ == BoundaryTag &&
                        _connection_tag_ == ConnectionTag) {
            _interface_container_ =
                InterfaceContainerType<_interface_tag_, _boundary_tag_,
                                       _connection_tag_>(
                    ngllz, ngllx, nquad_intersection, num_edges);
          }
        })
  }
};
} // namespace specfem::test_builder
