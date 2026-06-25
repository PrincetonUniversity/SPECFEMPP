#include "fixture.hpp"
#include "specfem/assembly.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/quadrature.hpp"

namespace specfem::test_configuration {

AssemblyMPI3D::AssemblyMPI3D(
    const specfem::mesh::mesh<specfem::element::dimension_tag::dim3> &mesh)
    : assembly_mesh([&mesh]() {
        const auto quadratures = []() {
          specfem::quadrature::gll::gll gll{};
          return specfem::quadrature::quadratures(gll);
        }();
        return specfem::assembly::mesh<specfem::element::dimension_tag::dim3>{
          mesh.nspec,
          mesh.control_nodes.ngnod,
          mesh.element_grid.ngllz,
          mesh.element_grid.nglly,
          mesh.element_grid.ngllx,
          mesh.tags,
          mesh.adjacency_graph,
          mesh.control_nodes,
          quadratures
        };
      }()),
      element_types(mesh.nspec, assembly_mesh.element_grid, assembly_mesh,
                    mesh.tags),
      fields(assembly_mesh, element_types, specfem::simulation::type::forward),
      mpi_interfaces(mesh.adjacency_graph, assembly_mesh, element_types,
                     specfem::simulation::type::forward, fields,
                     mesh.element_grid.ngllz, mesh.element_grid.nglly,
                     mesh.element_grid.ngllx) {}
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
  mesh = specfem::io::read_3d_mesh(mpi_database, specfem::attenuation::Setup{});

  assembly = specfem::test_configuration::AssemblyMPI3D(mesh);
}

void AssemblyMPI3DTest::TearDown() {
  // Any cleanup needed for each test
}
