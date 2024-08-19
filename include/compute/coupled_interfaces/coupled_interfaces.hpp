#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/coupled_interfaces/interface_container.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/specfem_enums.hpp"
#include "interface_container.hpp"
#include "mesh/coupled_interfaces/coupled_interfaces.hpp"

namespace specfem {
namespace compute {
struct coupled_interfaces {

  coupled_interfaces() = default;

  coupled_interfaces(
      const specfem::mesh::mesh &mesh, const specfem::compute::points &points,
      const specfem::compute::quadrature &quadrature,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties,
      const specfem::compute::mesh_to_compute_mapping &mapping);

  template <specfem::element::medium_tag medium1,
            specfem::element::medium_tag medium2>
  specfem::compute::interface_container<medium1, medium2>
  get_interface_container() const;

  specfem::compute::interface_container<specfem::element::medium_tag::elastic,
                                        specfem::element::medium_tag::acoustic>
      elastic_acoustic;

  specfem::compute::interface_container<
      specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::poroelastic>
      acoustic_poroelastic;

  specfem::compute::interface_container<
      specfem::element::medium_tag::elastic,
      specfem::element::medium_tag::poroelastic>
      elastic_poroelastic;
};
} // namespace compute
} // namespace specfem
