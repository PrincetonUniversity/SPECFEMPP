#include "mesh/coupled_interfaces/coupled_interfaces.hpp"
#include "mesh/coupled_interfaces/interface_container.hpp"
#include "mesh/coupled_interfaces/interface_container.tpp"

specfem::mesh::coupled_interfaces::coupled_interfaces::coupled_interfaces(
    std::ifstream &stream, const int num_interfaces_elastic_acoustic,
    const int num_interfaces_acoustic_poroelastic,
    const int num_interfaces_elastic_poroelastic, const specfem::MPI::MPI *mpi)
    : elastic_acoustic(num_interfaces_elastic_acoustic, stream, mpi),
      acoustic_poroelastic(num_interfaces_acoustic_poroelastic, stream, mpi),
      elastic_poroelastic(num_interfaces_elastic_poroelastic, stream, mpi) {
  return;
}

template <specfem::enums::element::type medium1,
          specfem::enums::element::type medium2>
std::variant<
    specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                       specfem::enums::element::type::acoustic>,
    specfem::mesh::interface_container<
        specfem::enums::element::type::acoustic,
        specfem::enums::element::type::poroelastic>,
    specfem::mesh::interface_container<
        specfem::enums::element::type::elastic,
        specfem::enums::element::type::poroelastic> >
specfem::mesh::coupled_interfaces::coupled_interfaces::get() const {
  if constexpr (medium1 == specfem::enums::element::type::elastic &&
                medium2 == specfem::enums::element::type::acoustic) {
    return elastic_acoustic;
  } else if constexpr (medium1 == specfem::enums::element::type::acoustic &&
                       medium2 == specfem::enums::element::type::poroelastic) {
    return acoustic_poroelastic;
  } else if constexpr (medium1 == specfem::enums::element::type::elastic &&
                       medium2 == specfem::enums::element::type::poroelastic) {
    return elastic_poroelastic;
  }
}

// Explicitly instantiate template class

template class specfem::mesh::interface_container<
    specfem::enums::element::type::elastic,
    specfem::enums::element::type::acoustic>;

template class specfem::mesh::interface_container<
    specfem::enums::element::type::acoustic,
    specfem::enums::element::type::poroelastic>;

template class specfem::mesh::interface_container<
    specfem::enums::element::type::elastic,
    specfem::enums::element::type::poroelastic>;

// Explicitly instantiate template member function
template int
specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                   specfem::enums::element::type::acoustic>::
    get_spectral_elem_index<specfem::enums::element::type::elastic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                   specfem::enums::element::type::acoustic>::
    get_spectral_elem_index<specfem::enums::element::type::acoustic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::enums::element::type::acoustic,
                                   specfem::enums::element::type::poroelastic>::
    get_spectral_elem_index<specfem::enums::element::type::acoustic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::enums::element::type::acoustic,
                                   specfem::enums::element::type::poroelastic>::
    get_spectral_elem_index<specfem::enums::element::type::poroelastic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                   specfem::enums::element::type::poroelastic>::
    get_spectral_elem_index<specfem::enums::element::type::elastic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                   specfem::enums::element::type::poroelastic>::
    get_spectral_elem_index<specfem::enums::element::type::poroelastic>(
        const int interface_index) const;

// Explicitly instantiate template member function
template std::variant<
    specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                       specfem::enums::element::type::acoustic>,
    specfem::mesh::interface_container<
        specfem::enums::element::type::acoustic,
        specfem::enums::element::type::poroelastic>,
    specfem::mesh::interface_container<
        specfem::enums::element::type::elastic,
        specfem::enums::element::type::poroelastic> >
specfem::mesh::coupled_interfaces::coupled_interfaces::get<
    specfem::enums::element::type::elastic,
    specfem::enums::element::type::acoustic>() const;

template std::variant<
    specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                       specfem::enums::element::type::acoustic>,
    specfem::mesh::interface_container<
        specfem::enums::element::type::acoustic,
        specfem::enums::element::type::poroelastic>,
    specfem::mesh::interface_container<
        specfem::enums::element::type::elastic,
        specfem::enums::element::type::poroelastic> >
specfem::mesh::coupled_interfaces::coupled_interfaces::get<
    specfem::enums::element::type::acoustic,
    specfem::enums::element::type::poroelastic>() const;

template std::variant<
    specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                       specfem::enums::element::type::acoustic>,
    specfem::mesh::interface_container<
        specfem::enums::element::type::acoustic,
        specfem::enums::element::type::poroelastic>,
    specfem::mesh::interface_container<
        specfem::enums::element::type::elastic,
        specfem::enums::element::type::poroelastic> >
specfem::mesh::coupled_interfaces::coupled_interfaces::get<
    specfem::enums::element::type::elastic,
    specfem::enums::element::type::poroelastic>() const;
