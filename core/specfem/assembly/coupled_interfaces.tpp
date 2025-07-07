#pragma once

#include "coupled_interfaces.hpp"
#include "enumerations/interface.hpp"
#include "coupled_interfaces/interface_container.hpp"
#include "coupled_interfaces/interface_container.tpp"

template <>
template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
specfem::assembly::interface_container<medium1, medium2>
specfem::assembly::coupled_interfaces<specfem::dimension::type::dim2>::get_interface_container() const {
  if constexpr (medium1 == specfem::element::medium_tag::elastic_psv &&
                medium2 == specfem::element::medium_tag::acoustic) {
    return elastic_acoustic;
  } else if constexpr (medium1 == specfem::element::medium_tag::acoustic &&
                       medium2 == specfem::element::medium_tag::elastic_psv) {
    return specfem::assembly::interface_container<medium1, medium2>(
        elastic_acoustic);
  } else if constexpr (medium1 == specfem::element::medium_tag::acoustic &&
                       medium2 == specfem::element::medium_tag::poroelastic) {
    return acoustic_poroelastic;
  } else if constexpr (medium1 == specfem::element::medium_tag::poroelastic &&
                       medium2 == specfem::element::medium_tag::acoustic) {
    return specfem::assembly::interface_container<medium1, medium2>(
        acoustic_poroelastic);
  } else if constexpr (medium1 == specfem::element::medium_tag::elastic_psv &&
                       medium2 == specfem::element::medium_tag::poroelastic) {
    return elastic_poroelastic;
  } else if constexpr (medium1 == specfem::element::medium_tag::poroelastic &&
                       medium2 == specfem::element::medium_tag::elastic_psv) {
    return specfem::assembly::interface_container<medium1, medium2>(
        elastic_poroelastic);
  }
}
