#include "../../SPECFEM_Environment.hpp"
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
#include "specfem/logger.hpp"
#include "specfem/timescheme.hpp"
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
  if (specfem::MPI_new::get_size() == 1) {
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

TEST(SEISMOGRAM_TESTS, acoustic_seismograms_test) {
  std::string config_filename = "seismogram/acoustic/test_config.yaml";

  specfem::MPI::MPI *mpi = SPECFEMEnvironment::get_mpi();

  test_config test_config = parse_test_config(config_filename, mpi);

  const std::string parameter_file = test_config.specfem_config;

  specfem::runtime_configuration::setup setup(parameter_file, __default_file__);

  const auto database_file = setup.get_databases();
  // std::cout << setup.print_header();

  // Set up GLL quadrature points
  const auto quadratures = setup.instantiate_quadrature();

  // Read mesh generated MESHFEM
  specfem::mesh::mesh mesh = specfem::io::read_2d_mesh(database_file, mpi);

  std::vector<std::shared_ptr<specfem::sources::source> > sources(0);

  const auto angle = setup.get_receiver_angle();
  const auto stations_node = setup.get_stations();
  auto receivers = specfem::io::read_2d_receivers(stations_node, angle);
  const auto stypes = setup.get_seismogram_types();

  specfem::assembly::assembly assembly(mesh, quadratures, sources, receivers,
                                       stypes, 0, 0, 0, 1,
                                       setup.get_simulation_type(), nullptr);

  const auto displacement_field = assembly.fields.forward.acoustic.h_field;
  const auto velocity_field = assembly.fields.forward.acoustic.h_field_dot;
  const auto acceleration_field =
      assembly.fields.forward.acoustic.h_field_dot_dot;

  const int nglob = assembly.fields.forward.nglob;

  read_field(test_config.displacement_field, displacement_field, nglob, 1);
  read_field(test_config.velocity_field, velocity_field, nglob, 1);
  read_field(test_config.acceleration_field, acceleration_field, nglob, 1);

  assembly.fields.copy_to_device();

  specfem::enums::element::quadrature::static_quadrature_points<5> qp5;

  specfem::domain::domain<
      specfem::wavefield::simulation_field::forward,
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::enums::element::quadrature::static_quadrature_points<5> >
      acoustic_domain_static(setup.get_dt(), assembly, qp5);

  acoustic_domain_static.compute_seismograms(0);

  assembly.receivers.sync_seismograms();

  type_real tol = 1e-5;

  std::vector<type_real> ground_truth = {
    0.0000000000000000,       -3.0960039922602422E-011,
    0.0000000000000000,       -8.3379189277645219E-011,
    4.4439509143591454E-010,  1.8368718019109921E-010,
    -3.8585618629325063E-010, 2.5444961269509465E-010,
    0.0000000000000000,       7.1296417833791251E-008,
    0.0000000000000000,       2.4165936811725470E-008,
    5.6330467200175704E-007,  3.5364862913830020E-007,
    -9.3482598617042963E-008, -3.6004231966844085E-007,
    0.0000000000000000,       2.0541840830987639E-005,
    0.0000000000000000,       9.0448554680095616E-006,
    -3.3644759982233034E-004, -3.5943211587533610E-004,
    3.1162494730438027E-004,  -2.9608074956535943E-004
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
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
