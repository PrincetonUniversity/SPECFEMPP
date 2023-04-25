#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "../utilities/include/compare_array.h"
#include "compute/interface.hpp"
#include "domain/interface.hpp"
#include "material.h"
#include "mesh.h"
#include "parameter_parser.h"
#include "quadrature.h"
#include "read_sources.h"
#include "solver/interface.hpp"
#include "timescheme/interface.hpp"
#include "utils.h"
#include "yaml-cpp/yaml.h"

// ----- Parse test config ------------- //

struct test_config {
  std::string specfem_config, displacement_field, velocity_field,
      acceleration_field;
};

void operator>>(const YAML::Node &Node, test_config &test_config) {
  test_config.specfem_config = Node["specfem_config"].as<std::string>();
  test_config.displacement_field = Node["displacement_field"].as<std::string>();
  test_config.velocity_field = Node["velocity_field"].as<std::string>();
  test_config.acceleration_field = Node["acceleration_field"].as<std::string>();

  return;
}

test_config parse_test_config(std::string test_configuration_file,
                              specfem::MPI::MPI *mpi) {

  YAML::Node yaml = YAML::LoadFile(test_configuration_file);
  const YAML::Node &tests = yaml["Tests"];
  const YAML::Node &serial = tests["serial"];

  test_config test_config;
  if (mpi->get_size() == 1) {
    serial >> test_config;
  }

  return test_config;
}

// ------------------------------------- //

// read field from fortran binary file
void read_field(
    const std::string filename,
    specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft> field,
    const int n1, const int n2) {

  assert(field.extent(0) == n1);
  assert(field.extent(1) == n2);

  std::ifstream stream;
  stream.open(filename);

  type_real ref_value;
  for (int i1 = 0; i1 < n1; i1++) {
    for (int i2 = 0; i2 < n2; i2++) {
      specfem::fortran_IO::fortran_read_line(stream, &ref_value);
      field(i1, i2) = ref_value;
    }
  }

  stream.close();

  return;
}

TEST(SEISMOGRAM_TESTS, elastic_seismograms_test) {
  std::string config_filename =
      "../../../tests/unittests/seismogram/test_config.yaml";

  specfem::MPI::MPI *mpi = MPIEnvironment::mpi_;

  test_config test_config = parse_test_config(config_filename, mpi);

  const std::string parameter_file = test_config.specfem_config;

  specfem::runtime_configuration::setup setup(parameter_file);

  const auto [database_file, sources_file] = setup.get_databases();
  // mpi->cout(setup.print_header());

  // Set up GLL quadrature points
  auto [gllx, gllz] = setup.instantiate_quadrature();

  const auto angle = setup.get_receiver_angle();
  const auto stations_filename = setup.get_stations_file();
  auto receivers = specfem::read_receivers(stations_filename, angle);

  // Read mesh generated MESHFEM
  std::vector<specfem::material *> materials;
  specfem::mesh mesh(database_file, materials, mpi);

  // Generate compute structs to be used by the solver
  specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                    gllz);
  specfem::compute::partial_derivatives partial_derivatives(
      mesh.coorg, mesh.material_ind.knods, gllx, gllz);
  specfem::compute::properties material_properties(mesh.material_ind.kmato,
                                                   materials, mesh.nspec,
                                                   gllx.get_N(), gllz.get_N());

  // locate the recievers
  for (auto &receiver : receivers)
    receiver->locate(compute.coordinates.coord, compute.h_ibool, gllx.get_hxi(),
                     gllz.get_hxi(), mesh.nproc, mesh.coorg,
                     mesh.material_ind.knods, mesh.npgeo,
                     material_properties.h_ispec_type, mpi);

  // Setup solver compute struct

  const type_real xmax = compute.coordinates.xmax;
  const type_real xmin = compute.coordinates.xmin;
  const type_real zmax = compute.coordinates.zmax;
  const type_real zmin = compute.coordinates.zmin;

  const auto stypes = setup.get_seismogram_types();

  specfem::compute::receivers compute_receivers(receivers, stypes, gllx, gllz,
                                                xmax, xmin, zmax, zmin, 1, mpi);

  const int nglob = specfem::utilities::compute_nglob(compute.h_ibool);
  specfem::Domain::Domain *domain = new specfem::Domain::Elastic(
      2, nglob, &compute, &material_properties, &partial_derivatives, NULL,
      &compute_receivers, &gllx, &gllz);

  const auto displacement_field = domain->get_host_field();
  const auto velocity_field = domain->get_host_field_dot();
  const auto acceleration_field = domain->get_host_field_dot_dot();

  read_field(test_config.displacement_field, displacement_field, nglob, 2);
  read_field(test_config.velocity_field, velocity_field, nglob, 2);
  read_field(test_config.acceleration_field, acceleration_field, nglob, 2);

  domain->sync_field(specfem::sync::HostToDevice);
  domain->sync_field_dot(specfem::sync::HostToDevice);
  domain->sync_field_dot_dot(specfem::sync::HostToDevice);

  domain->compute_seismogram(0);

  compute_receivers.sync_seismograms();

  type_real tol = 1e-6;

  std::vector<type_real> ground_truth = {
    2.0550622561421793e-032,  1.8686745381396431e-032,
    1.3080300305168132e-030,  -2.1906492252753700e-032,
    -9.0217861369207779e-027, -3.9905076983364219e-013,
    -1.4026049322672686e-032, 1.6751417898470163e-019,
    1.8389801799426992e-029,  1.4203556224296551e-029,
    2.2285583550372299e-027,  -1.5172766331841327e-027,
    -4.5863248350600626e-023, 3.2148834521809124e-009,
    -3.4039863284108916e-029, -9.4081223795340489e-016,
    -6.8297735413853655e-028, -5.6148816325380543e-027,
    2.3802139708429514e-024,  -2.7232009472557120e-024,
    4.8572257327350591e-021,  -6.7130089207176591e-007,
    -2.3332620537792063e-026, -9.3957047271406868e-013
  };

  int index = 0;

  for (int isys = 0; isys < stypes.size(); isys++) {
    for (int irec = 0; irec < receivers.size(); irec++) {
      for (int idim = 0; idim < 2; idim++) {
        EXPECT_NEAR(compute_receivers.h_seismogram(0, isys, irec, idim),
                    ground_truth[index], std::fabs(tol * ground_truth[index]));
        index++;
      }
    }
  }

  assert(index == ground_truth.size());

  return;
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
