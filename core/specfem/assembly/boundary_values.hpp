#pragma once

#include "boundaries.hpp"
#include "boundary_values/boundary_medium_container.hpp"
#include "boundary_values/boundary_values_container.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh.hpp"
#include "properties.hpp"

namespace specfem::assembly {
class boundary_values {
public:
  boundary_values() = default;

  specfem::assembly::boundary_value_container<
      specfem::dimension::type::dim2, specfem::element::boundary_tag::stacey>
      stacey;

  specfem::assembly::boundary_value_container<
      specfem::dimension::type::dim2,
      specfem::element::boundary_tag::composite_stacey_dirichlet>
      composite_stacey_dirichlet;

  boundary_values(const int nstep, const specfem::assembly::mesh mesh,
                  const specfem::assembly::element_types element_types,
                  const specfem::assembly::boundaries boundaries);

  template <specfem::element::boundary_tag BoundaryTag>
  specfem::assembly::boundary_value_container<specfem::dimension::type::dim2,
                                              BoundaryTag>
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
