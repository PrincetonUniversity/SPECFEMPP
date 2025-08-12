#include <gtest/gtest.h>
#include <memory>
#include <tuple>

#include "io/interface.hpp"
#include "mortar/dg/interface_container/containers_test.hpp"
#include "mortar/fixture/mortar_fixtures.hpp"

#include "../../../MPI_environment.hpp"
#include "containers_test.tpp"
#include "quadrature/quadratures.hpp"
#include "specfem/receivers.hpp"

static constexpr int NGLL = 5;
/* ==============
 * Container test
 * ==============
 *
 * This test validates interface_modules:
 *  - single_edge_container
 *    - ~~acoustic~~
 *          [!] no mesh
 *    - elastic
 *  - double_edge_container
 *    - ~~acoustic-acoustic~~
 *          [!] no mesh
 *    - ~~acoustic-elastic~~
 *          [!] no mesh
 *    - ~~elastic-acoustic~~
 *          [!] no mesh
 *    - elastic-elastic
 *  - ~~interface / geometry containers~~
 *      [!] containers not yet implemented
 * for a non-exhaustive collection of permutations of media.
 *
 * modules are tested independently of the assembly through the test containers
 * in `dg/interface_container/test_containers.hpp`. The following are verfied:
 *  - correct point access with EdgePolicy and double_edge_container
 *      [TODO]
 *  - correctly initializes and retrieves mortar transfer tensor
 *      [!] containers not yet implemented
 *      [!] no nonconforming mesh implemented
 *  - correctly initializes and retrieves edge normal
 *      [!] containers not yet implemented
 */
void test_container(const test_configuration::mesh &mesh_config) {
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  auto &mesh = mesh_config.get_mesh();
  specfem::assembly::assembly<specfem::dimension::type::dim2> assembly(
      mesh,
      specfem::quadrature::quadratures(
          specfem::quadrature::gll::gll(0, 0, NGLL)),
      std::vector<std::shared_ptr<specfem::sources::source> >(),
      std::vector<std::shared_ptr<
          specfem::receivers::receiver<specfem::dimension::type::dim2> > >(),
      std::vector<specfem::wavefield::type>(), 0, 1, 1, 100, 1,
      specfem::simulation::type::forward, false, nullptr);

  test_configuration::interface_containers::test_on_mesh(mesh_config, mesh,
                                                         assembly);
}

TEST_F(MESHES, interface_containers) {
  for (const auto &mesh : *this) {
    if (!(mesh.interface_fluid_2d || mesh.interface_fluid_fluid_2d ||
          mesh.interface_fluid_solid_2d || mesh.interface_solid_2d ||
          mesh.interface_solid_fluid_2d || mesh.interface_solid_solid_2d)) {
      std::cout << "'interface container' skipped for " << mesh.name
                << std::endl;
      continue;
    }
    try {
      test_container(mesh);
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m interface container -- "
                << mesh.name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
    } catch (std::exception &e) {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;31m[FAILED]\033[0m \n"
                << "-------------------------------------------------------\n"
                << "- Test: " << mesh.name << "\n"
                << "- Error: " << e.what() << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      ADD_FAILURE();
    }
  }
}
