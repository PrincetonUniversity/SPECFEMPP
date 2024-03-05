#ifndef _COUPLED_INTERFACES_HPP_
#define _COUPLED_INTERFACES_HPP_

// #include "acoustic_poroelastic.hpp"
// #include "elastic_acoustic.hpp"
// #include "elastic_poroelastic.hpp"
#include "enumerations/specfem_enums.hpp"
#include "interface_container.hpp"
#include "specfem_mpi/interface.hpp"
#include <variant>

namespace specfem {
namespace mesh {

struct coupled_interfaces {
public:
  coupled_interfaces()
      : elastic_acoustic(), acoustic_poroelastic(), elastic_poroelastic(){};
  coupled_interfaces(std::ifstream &stream,
                     const int num_interfaces_elastic_acoustic,
                     const int num_interfaces_acoustic_poroelastic,
                     const int num_interfaces_elastic_poroelastic,
                     const specfem::MPI::MPI *mpi);

  template <specfem::element::medium_tag medium1,
            specfem::element::medium_tag medium2>
  std::variant<specfem::mesh::interface_container<
                   specfem::element::medium_tag::elastic,
                   specfem::element::medium_tag::acoustic>,
               specfem::mesh::interface_container<
                   specfem::element::medium_tag::acoustic,
                   specfem::element::medium_tag::poroelastic>,
               specfem::mesh::interface_container<
                   specfem::element::medium_tag::elastic,
                   specfem::element::medium_tag::poroelastic> >
  get() const;

  specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                     specfem::element::medium_tag::acoustic>
      elastic_acoustic;

  specfem::mesh::interface_container<specfem::element::medium_tag::acoustic,
                                     specfem::element::medium_tag::poroelastic>
      acoustic_poroelastic;

  specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                     specfem::element::medium_tag::poroelastic>
      elastic_poroelastic;
};

} // namespace mesh
} // namespace specfem
#endif /* _COUPLED_INTERFACES_HPP_ */
