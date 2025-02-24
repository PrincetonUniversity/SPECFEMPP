#ifndef _COMPUTE_COUPLED_INTERFACES_COUPLED_INTERFACES_TPP
#define _COMPUTE_COUPLED_INTERFACES_COUPLED_INTERFACES_TPP

#include "coupled_interfaces.hpp"
#include "enumerations/specfem_enums.hpp"
#include "interface_container.hpp"
#include "interface_container.tpp"

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
specfem::compute::interface_container<medium1, medium2>
specfem::compute::coupled_interfaces::get_interface_container() const {
  if constexpr (medium1 == specfem::element::medium_tag::elastic_sv &&
                medium2 == specfem::element::medium_tag::acoustic) {
    return elastic_acoustic;
  } else if constexpr (medium1 == specfem::element::medium_tag::acoustic &&
                       medium2 == specfem::element::medium_tag::elastic_sv) {
    return specfem::compute::interface_container<medium1, medium2>(
        elastic_acoustic);
  } else if constexpr (medium1 == specfem::element::medium_tag::acoustic &&
                       medium2 == specfem::element::medium_tag::poroelastic) {
    return acoustic_poroelastic;
  } else if constexpr (medium1 == specfem::element::medium_tag::poroelastic &&
                       medium2 == specfem::element::medium_tag::acoustic) {
    return specfem::compute::interface_container<medium1, medium2>(
        acoustic_poroelastic);
  } else if constexpr (medium1 == specfem::element::medium_tag::elastic_sv &&
                       medium2 == specfem::element::medium_tag::poroelastic) {
    return elastic_poroelastic;
  } else if constexpr (medium1 == specfem::element::medium_tag::poroelastic &&
                       medium2 == specfem::element::medium_tag::elastic_sv) {
    return specfem::compute::interface_container<medium1, medium2>(
        elastic_poroelastic);
  }
}

#endif
