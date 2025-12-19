#pragma once

#include "dim2/impl/boundary_value_container.hpp"
#include "dim3/impl/boundary_value_container.hpp"
#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/properties.hpp"

namespace specfem::assembly {

template <specfem::dimension::type DimensionTag> class boundary_values {
public:
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag
  boundary_values() = default;

  specfem::assembly::boundary_values_impl::boundary_value_container<
      dimension_tag, specfem::element::boundary_tag::stacey>
      stacey;

  specfem::assembly::boundary_values_impl::boundary_value_container<
      dimension_tag, specfem::element::boundary_tag::composite_stacey_dirichlet>
      composite_stacey_dirichlet;

  boundary_values(
      const int nstep, const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::boundaries<dimension_tag> &boundaries);

  template <specfem::element::boundary_tag BoundaryTag>
  specfem::assembly::boundary_values_impl::boundary_value_container<
      dimension_tag, BoundaryTag>
  get_container() const {
    if constexpr (BoundaryTag == specfem::element::boundary_tag::stacey) {
      return stacey;
    } else if constexpr (BoundaryTag == specfem::element::boundary_tag::
                                            composite_stacey_dirichlet) {
      return composite_stacey_dirichlet;
    } else {
      return {};
    }
  }

  void copy_to_host() {
    stacey.sync_to_host();
    composite_stacey_dirichlet.sync_to_host();
  }

  void copy_to_device() {
    stacey.sync_to_device();
    composite_stacey_dirichlet.sync_to_device();
  }
};
} // namespace specfem::assembly
