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

  template <specfem::enums::element::type medium1,
            specfem::enums::element::type medium2>
  std::variant<specfem::mesh::interface_container<
                   specfem::enums::element::type::elastic,
                   specfem::enums::element::type::acoustic>,
               specfem::mesh::interface_container<
                   specfem::enums::element::type::acoustic,
                   specfem::enums::element::type::poroelastic>,
               specfem::mesh::interface_container<
                   specfem::enums::element::type::elastic,
                   specfem::enums::element::type::poroelastic> >
  get() const;

  specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                     specfem::enums::element::type::acoustic>
      elastic_acoustic;

  specfem::mesh::interface_container<specfem::enums::element::type::acoustic,
                                     specfem::enums::element::type::poroelastic>
      acoustic_poroelastic;

  specfem::mesh::interface_container<specfem::enums::element::type::elastic,
                                     specfem::enums::element::type::poroelastic>
      elastic_poroelastic;
};

} // namespace mesh
} // namespace specfem
#endif /* _COUPLED_INTERFACES_HPP_ */
