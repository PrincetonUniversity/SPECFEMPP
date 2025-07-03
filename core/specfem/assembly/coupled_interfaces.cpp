#include "coupled_interfaces.hpp"
#include "coupled_interfaces.tpp"
#include "coupled_interfaces/interface_container.hpp"
#include "coupled_interfaces/interface_container.tpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "mesh/mesh.hpp"

// // Topological map ordering for coupled elements

// // +-----------------+      +-----------------+
// // |                 |      |                 |
// // |               R | ^  ^ | L               |
// // |               I | |  | | E               |
// // |               G | |  | | F               |
// // |               H | |  | | T               |
// // |               T | |  | |                 |
// // |     BOTTOM      |      |      BOTTOM     |
// // +-----------------+      +-----------------+
// //   -------------->          -------------->
// //   -------------->          -------------->
// // +-----------------+      +-----------------+
// // |      TOP        |      |       TOP       |
// // |               R | ^  ^ | L               |
// // |               I | |  | | E               |
// // |               G | |  | | F               |
// // |               H | |  | | T               |
// // |               T | |  | |                 |
// // |                 |      |                 |
// // +-----------------+      +-----------------+

specfem::assembly::coupled_interfaces::coupled_interfaces(
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::mesh<specfem::dimension::type::dim2>
        &mesh_assembly,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    const specfem::assembly::element_types &element_types)
    : elastic_acoustic(mesh, mesh_assembly, jacobian_matrix, element_types),
      elastic_poroelastic(mesh, mesh_assembly, jacobian_matrix, element_types),
      acoustic_poroelastic(mesh, mesh_assembly, jacobian_matrix,
                           element_types) {}
// Explicit template instantiation

template class specfem::assembly::interface_container<
    specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::acoustic>;

template class specfem::assembly::interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>;

template class specfem::assembly::interface_container<
    specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::poroelastic>;

template specfem::assembly::interface_container<
    specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::acoustic>
specfem::assembly::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::acoustic>() const;

template specfem::assembly::interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic_psv>
specfem::assembly::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic_psv>() const;

template specfem::assembly::interface_container<
    specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::poroelastic>
specfem::assembly::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::poroelastic>() const;

template specfem::assembly::interface_container<
    specfem::element::medium_tag::poroelastic,
    specfem::element::medium_tag::elastic_psv>
specfem::assembly::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::poroelastic,
    specfem::element::medium_tag::elastic_psv>() const;

template specfem::assembly::interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>
specfem::assembly::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>() const;

template specfem::assembly::interface_container<
    specfem::element::medium_tag::poroelastic,
    specfem::element::medium_tag::acoustic>
specfem::assembly::coupled_interfaces::get_interface_container<
    specfem::element::medium_tag::poroelastic,
    specfem::element::medium_tag::acoustic>() const;

// Explicit template member function instantiation
