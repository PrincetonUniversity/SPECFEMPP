#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/compare_array.h"
#include "compute/interface.hpp"
#include "constants.hpp"
#include "domain/interface.hpp"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "receiver/interface.hpp"
#include "solver/interface.hpp"
#include "timescheme/interface.hpp"
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
      "../../../tests/unit-tests/seismogram/acoustic/test_config.yaml";

  specfem::MPI::MPI *mpi = MPIEnvironment::mpi_;

  test_config test_config = parse_test_config(config_filename, mpi);

  const std::string parameter_file = test_config.specfem_config;

  specfem::runtime_configuration::setup setup(parameter_file, __default_file__);

  const auto [database_file, sources_file] = setup.get_databases();
  // mpi->cout(setup.print_header());

  // Set up GLL quadrature points
  auto [gllx, gllz] = setup.instantiate_quadrature();

  const auto angle = setup.get_receiver_angle();
  const auto stations_filename = setup.get_stations_file();
  auto receivers = specfem::receivers::read_receivers(stations_filename, angle);

  // Read mesh generated MESHFEM
  std::vector<std::shared_ptr<specfem::material::material> > materials;
  specfem::mesh::mesh mesh(database_file, materials, mpi);

  // Generate compute structs to be used by the solver
  specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                    gllz);
  specfem::compute::partial_derivatives partial_derivatives(
      mesh.coorg, mesh.material_ind.knods, gllx, gllz);
  specfem::compute::properties material_properties(
      mesh.material_ind.kmato, materials, mesh.nspec, gllx->get_N(),
      gllz->get_N());

  specfem::compute::boundaries boundary_conditions(
      mesh.material_ind.kmato, materials, mesh.acfree_surface,
      mesh.abs_boundary);

  // locate the recievers
  for (auto &receiver : receivers)
    receiver->locate(compute.coordinates.coord, compute.h_ibool,
                     gllx->get_hxi(), gllz->get_hxi(), mesh.nproc, mesh.coorg,
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
  specfem::enums::element::quadrature::static_quadrature_points<5> qp5;
  specfem::domain::domain<
      specfem::enums::element::medium::acoustic,
      specfem::enums::element::quadrature::static_quadrature_points<5> >
      acoustic_domain_static(nglob, qp5, &compute, material_properties,
                             partial_derivatives, boundary_conditions,
                             specfem::compute::sources(), compute_receivers,
                             gllx, gllz);

  const auto displacement_field = acoustic_domain_static.get_host_field();
  const auto velocity_field = acoustic_domain_static.get_host_field_dot();
  const auto acceleration_field =
      acoustic_domain_static.get_host_field_dot_dot();

  read_field(test_config.displacement_field, displacement_field, nglob, 1);
  read_field(test_config.velocity_field, velocity_field, nglob, 1);
  read_field(test_config.acceleration_field, acceleration_field, nglob, 1);

  acoustic_domain_static.sync_field(specfem::sync::HostToDevice);
  acoustic_domain_static.sync_field_dot(specfem::sync::HostToDevice);
  acoustic_domain_static.sync_field_dot_dot(specfem::sync::HostToDevice);

  acoustic_domain_static.compute_seismogram(0);

  compute_receivers.sync_seismograms();

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
