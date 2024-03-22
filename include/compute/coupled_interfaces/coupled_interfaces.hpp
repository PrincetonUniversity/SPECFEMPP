#ifndef _COMPUTE_COUPLED_INTERFACES_HPP
#define _COMPUTE_COUPLED_INTERFACES_HPP

#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/specfem_enums.hpp"
#include "interface_container.hpp"
#include "mesh/coupled_interfaces/coupled_interfaces.hpp"

namespace specfem {
namespace compute {
struct coupled_interfaces {

  coupled_interfaces() = default;

  coupled_interfaces(
      const specfem::compute::mesh &mesh,
      const specfem::compute::properties &properties,
      const specfem::mesh::coupled_interfaces &coupled_interfaces);

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

#endif
