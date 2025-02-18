#include "IO/mesh/impl/fortran/dim2/read_interfaces.hpp"
#include "IO/fortranio/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
specfem::mesh::interface_container<DimensionType, medium1, medium2>
specfem::IO::mesh::impl::fortran::dim2::read_interfaces(
    const int num_interfaces, std::ifstream &stream,
    const specfem::MPI::MPI *mpi) {

  specfem::mesh::interface_container<DimensionType, medium1, medium2> interface(
      num_interfaces);

  if (!num_interfaces)
    return interface;

  int medium1_ispec_l, medium2_ispec_l;

  for (int i = 0; i < num_interfaces; i++) {
    specfem::IO::fortran_read_line(stream, &medium2_ispec_l, &medium1_ispec_l);
    interface.medium1_index_mapping(i) = medium1_ispec_l - 1;
    interface.medium2_index_mapping(i) = medium2_ispec_l - 1;
  }

  return interface;
}

// Explicit instantiation of the template function for the different medium
// interfaces elastic/acoustic
template specfem::mesh::interface_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>
specfem::IO::mesh::impl::fortran::dim2::read_interfaces<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>(const int num_interfaces,
                                            std::ifstream &stream,
                                            const specfem::MPI::MPI *mpi);

// acoustic/poroelastic
template specfem::mesh::interface_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>
specfem::IO::mesh::impl::fortran::dim2::read_interfaces<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>(const int num_interfaces,
                                               std::ifstream &stream,
                                               const specfem::MPI::MPI *mpi);

// elastic/poroelastic
template specfem::mesh::interface_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::poroelastic>
specfem::IO::mesh::impl::fortran::dim2::read_interfaces<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::poroelastic>(const int num_interfaces,
                                               std::ifstream &stream,
                                               const specfem::MPI::MPI *mpi);

specfem::mesh::coupled_interfaces<specfem::dimension::type::dim2>
specfem::IO::mesh::impl::fortran::dim2::read_coupled_interfaces(
    std::ifstream &stream, const int num_interfaces_elastic_acoustic,
    const int num_interfaces_acoustic_poroelastic,
    const int num_interfaces_elastic_poroelastic,
    const specfem::MPI::MPI *mpi) {

  auto elastic_acoustic =
      specfem::IO::mesh::impl::fortran::dim2::read_interfaces<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::medium_tag::acoustic>(
          num_interfaces_elastic_acoustic, stream, mpi);

  auto acoustic_poroelastic =
      specfem::IO::mesh::impl::fortran::dim2::read_interfaces<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::poroelastic>(
          num_interfaces_acoustic_poroelastic, stream, mpi);

  auto elastic_poroelastic =
      specfem::IO::mesh::impl::fortran::dim2::read_interfaces<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::medium_tag::poroelastic>(
          num_interfaces_elastic_poroelastic, stream, mpi);

  return specfem::mesh::coupled_interfaces<specfem::dimension::type::dim2>(
      elastic_acoustic, acoustic_poroelastic, elastic_poroelastic);
}
