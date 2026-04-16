#include "fixture.hpp"
#include "specfem/assembly.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/quadrature.hpp"

namespace specfem::test_configuration {

AssemblyMPI3D::AssemblyMPI3D(
    const specfem::mesh::mesh<specfem::element::dimension_tag::dim3> &mesh)
    : mpi_interfaces([&mesh]() {
        const int nspec = mesh.nspec;
        const int ngnod = mesh.control_nodes.ngnod;
        const int ngllz = mesh.element_grid.ngllz;
        const int nglly = mesh.element_grid.nglly;
        const int ngllx = mesh.element_grid.ngllx;
        const auto quadratures = []() {
          specfem::quadrature::gll::gll gll{};
          return specfem::quadrature::quadratures(gll);
        }();

        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            assembly_mesh{ nspec,
                           ngnod,
                           ngllz,
                           nglly,
                           ngllx,
                           mesh.tags,
                           mesh.adjacency_graph,
                           mesh.control_nodes,
                           quadratures };

        specfem::assembly::mpi<specfem::element::dimension_tag::dim3>
            mpi_interfaces(mesh.adjacency_graph, assembly_mesh,
                           mesh.element_grid.ngllz, mesh.element_grid.nglly,
                           mesh.element_grid.ngllx);

        return mpi_interfaces;
      }()) {}
} // namespace specfem::test_configuration

void AssemblyMPI3DTest::SetUp() {
  // Check if MPI size requirement was met during environment setup
  if (!SPECFEMEnvironment::IsMPISizeValid()) {
    GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();
  }

  // Check if the current rank is within the participating range for this test
  if (!specfem::MPI::is_active()) {
    GTEST_SKIP() << "Test designed for 4 processes. Rank "
                 << specfem::MPI::get_rank()
                 << " is outside the participating range [0-3].";
  }

  const auto &folder = GetParam();
  const std::string database = "data/mpi/dim3/" + folder + "/Database.bin";
  const auto mpi_database = specfem::MPI::format_proc_filename(database);
  mesh =
      specfem::io::read_3d_mesh(mpi_database, false /*attenuation disabled*/);

  assembly = specfem::test_configuration::AssemblyMPI3D(mesh);
}

void AssemblyMPI3DTest::TearDown() {
  // Any cleanup needed for each test
}
