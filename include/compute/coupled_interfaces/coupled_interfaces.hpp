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

  specfem::compute::interface_container<specfem::enums::element::type::elastic,
                                        specfem::enums::element::type::acoustic>
      elastic_acoustic;

  specfem::compute::interface_container<
      specfem::enums::element::type::acoustic,
      specfem::enums::element::type::poroelastic>
      acoustic_poroelastic;

  specfem::compute::interface_container<
      specfem::enums::element::type::elastic,
      specfem::enums::element::type::poroelastic>
      elastic_poroelastic;
};
} // namespace compute
} // namespace specfem

#endif
