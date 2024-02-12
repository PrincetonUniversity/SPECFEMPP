#ifndef _COMPUTE_COUPLED_INTERFACES_COUPLED_INTERFACES_TPP
#define _COMPUTE_COUPLED_INTERFACES_COUPLED_INTERFACES_TPP

#include "coupled_interfaces.hpp"
#include "enumerations/specfem_enums.hpp"
#include "interface_container.hpp"
#include "interface_container.tpp"

template <specfem::enums::element::type medium1,
          specfem::enums::element::type medium2>
specfem::compute::interface_container<medium1, medium2>
specfem::compute::coupled_interfaces::get_interface_container() const {
  if constexpr (medium1 == specfem::enums::element::type::elastic &&
                medium2 == specfem::enums::element::type::acoustic) {
    return elastic_acoustic;
  } else if constexpr (medium1 == specfem::enums::element::type::acoustic &&
                       medium2 == specfem::enums::element::type::elastic) {
    return specfem::compute::interface_container<medium1, medium2>(
        elastic_acoustic);
  } else if constexpr (medium1 == specfem::enums::element::type::acoustic &&
                       medium2 == specfem::enums::element::type::poroelastic) {
    return acoustic_poroelastic;
  } else if constexpr (medium1 == specfem::enums::element::type::poroelastic &&
                       medium2 == specfem::enums::element::type::acoustic) {
    return specfem::compute::interface_container<medium1, medium2>(
        acoustic_poroelastic);
  } else if constexpr (medium1 == specfem::enums::element::type::elastic &&
                       medium2 == specfem::enums::element::type::poroelastic) {
    return elastic_poroelastic;
  } else if constexpr (medium1 == specfem::enums::element::type::poroelastic &&
                       medium2 == specfem::enums::element::type::elastic) {
    return specfem::compute::interface_container<medium1, medium2>(
        elastic_poroelastic);
  }
}

#endif
