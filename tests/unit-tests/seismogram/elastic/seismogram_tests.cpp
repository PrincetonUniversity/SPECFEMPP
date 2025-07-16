#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
// #include "../../utilities/include/compare_array.h"
#include "constants.hpp"
#include "domain/domain.hpp"
#include "io/fortranio/interface.hpp"
#include "io/interface.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "solver/solver.hpp"
#include "specfem/assembly.hpp"
#include "timescheme/timescheme.hpp"
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
      specfem::io::fortran_read_line(stream, &ref_value);
      field(i1, i2) = ref_value;
    }
  }

  stream.close();

  return;
}

TEST(SEISMOGRAM_TESTS, elastic_seismograms_test) {
  std::string config_filename = "seismogram/elastic/test_config.yaml";

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  test_config test_config = parse_test_config(config_filename, mpi);

  const std::string parameter_file = test_config.specfem_config;

  specfem::runtime_configuration::setup setup(parameter_file, __default_file__);

  const auto database_file = setup.get_databases();
  // mpi->cout(setup.print_header());

  // Set up GLL quadrature points
  const auto quadratures = setup.instantiate_quadrature();

  // Read mesh generated MESHFEM
  specfem::mesh::mesh mesh = specfem::io::read_2d_mesh(database_file, mpi);

  std::vector<std::shared_ptr<specfem::sources::source> > sources(0);

  const auto angle = setup.get_receiver_angle();
  const auto stations_node = setup.get_stations();
  auto receivers = specfem::io::read_receivers(stations_node, angle);
  const auto stypes = setup.get_seismogram_types();

  specfem::assembly::assembly assembly(mesh, quadratures, sources, receivers,
                                       stypes, 0, 0, 0, 1,
                                       setup.get_simulation_type(), nullptr);

  const auto displacement_field = assembly.fields.forward.elastic.h_field;
  const auto velocity_field = assembly.fields.forward.elastic.h_field_dot;
  const auto acceleration_field =
      assembly.fields.forward.elastic.h_field_dot_dot;

  const int nglob = assembly.fields.forward.nglob;

  read_field(test_config.displacement_field, displacement_field, nglob, 2);
  read_field(test_config.velocity_field, velocity_field, nglob, 2);
  read_field(test_config.acceleration_field, acceleration_field, nglob, 2);

  assembly.fields.copy_to_device();

  specfem::enums::element::quadrature::static_quadrature_points<5> qp5;

  specfem::domain::domain<
      specfem::wavefield::simulation_field::forward,
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
      specfem::enums::element::quadrature::static_quadrature_points<5> >
      elastic_domain_static(setup.get_dt(), assembly, qp5);

  elastic_domain_static.compute_seismograms(0);

  assembly.receivers.sync_seismograms();

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
        EXPECT_NEAR(assembly.receivers.h_seismogram(0, isys, irec, idim),
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
