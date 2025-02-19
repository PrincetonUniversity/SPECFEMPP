#include "archive/stiffness1.hpp"
#include "execute.hpp"
#include "stiffness.hpp"
// #include "divide.hpp"

namespace specfem {
namespace benchmarks {

template <bool flag>
void benchmark(specfem::compute::assembly &assembly,
               std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme) {
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto isotropic = specfem::element::property_tag::isotropic;
  constexpr auto anisotropic = specfem::element::property_tag::anisotropic;

  const int nstep = time_scheme->get_max_timestep();

  const auto solver_start_time = std::chrono::system_clock::now();

  // const auto &field = assembly.fields.get_simulation_field<wavefield>();
  // std::cout << ">>>>" << field.acoustic.nglob << " | " << field.elastic.nglob
  // << std::endl;

  // specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft>
  // acoustic_field("acoustic_field", field.acoustic.nglob/ngll/ngll,ngll,ngll,
  // field.acoustic.components); specfem::kokkos::DeviceView4d<type_real,
  // Kokkos::LayoutLeft> acoustic_field_dot_dot("acoustic_field_dot_dot",
  // field.acoustic.nglob/ngll/ngll,ngll,ngll, field.acoustic.components);

  // specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft>
  // elastic_field("elastic_field", field.elastic.nglob/ngll/ngll,ngll,ngll,
  // field.elastic.components); specfem::kokkos::DeviceView4d<type_real,
  // Kokkos::LayoutLeft> elastic_field_dot_dot("elastic_field_dot_dot",
  // field.elastic.nglob/ngll/ngll,ngll,ngll, field.elastic.components);

  // specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  // acoustic_field("acoustic_field", field.acoustic.nglob,
  // field.acoustic.components); specfem::kokkos::DeviceView2d<type_real,
  // Kokkos::LayoutLeft> acoustic_field_dot_dot("acoustic_field_dot_dot",
  // field.acoustic.nglob, field.acoustic.components);

  // specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  // elastic_field("elastic_field", field.elastic.nglob,
  // field.elastic.components); specfem::kokkos::DeviceView2d<type_real,
  // Kokkos::LayoutLeft> elastic_field_dot_dot("elastic_field_dot_dot",
  // field.elastic.nglob, field.elastic.components);

  for (const auto [istep, dt] : time_scheme->iterate_forward()) {
    compute_stiffness_interaction<acoustic, isotropic, flag>(assembly, istep);
    compute_stiffness_interaction<elastic, isotropic, flag>(assembly, istep);
    compute_stiffness_interaction<elastic, anisotropic, flag>(assembly, istep);

    // compute_stiffness_interaction2<acoustic, isotropic, flag>(assembly,
    // istep); compute_stiffness_interaction2<elastic, isotropic,
    // flag>(assembly, istep); compute_stiffness_interaction2<elastic,
    // anisotropic, flag>(assembly, istep);

    // if constexpr (flag) {
    //   compute_stiffness_interaction<acoustic, isotropic, false>(assembly,
    //   istep); compute_stiffness_interaction<elastic, isotropic,
    //   false>(assembly, istep); compute_stiffness_interaction<elastic,
    //   anisotropic, false>(assembly, istep);
    // } else {
    //   compute_stiffness_interaction2<acoustic, isotropic, false>(assembly,
    //   istep); compute_stiffness_interaction2<elastic, isotropic,
    //   false>(assembly, istep); compute_stiffness_interaction2<elastic,
    //   anisotropic, false>(assembly, istep);
    // }

    // divide_mass_matrix<dimension, wavefield, acoustic>(assembly);
    // divide_mass_matrix<dimension, wavefield, elastic>(assembly);

    // if ((istep + 1) % 400 == 0) {
    //   std::cout << "Progress : executed " << istep + 1 << " steps of " <<
    //   nstep
    //             << " steps" << std::endl;
    // }
  }

  const auto solver_end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> solver_time =
      solver_end_time - solver_start_time;
  std::cout << " " << solver_time.count() << "s" << std::endl;
}

void run_benchmark(const YAML::Node &parameter_dict,
                   const YAML::Node &default_dict, specfem::MPI::MPI *mpi) {

  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------
  auto start_time = std::chrono::system_clock::now();
  specfem::runtime_configuration::setup setup(parameter_dict, default_dict);
  const auto database_filename = setup.get_databases();

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();
  const auto mesh = specfem::IO::read_mesh(database_filename, mpi);
  // specfem::IO::print_mesh(mesh, mpi);

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Timescheme
  // --------------------------------------------------------------
  const auto time_scheme = setup.instantiate_timescheme();

  const int max_seismogram_time_step = time_scheme->get_max_seismogram_step();

  const int nstep_between_samples = time_scheme->get_nstep_between_samples();
  const type_real dt = setup.get_dt();
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Generate Assembly
  // --------------------------------------------------------------
  // mpi->cout("-------------------------------");
  const std::vector<std::shared_ptr<specfem::sources::source> > sources;
  const std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers;
  const int nsteps = setup.get_nsteps();
  specfem::compute::assembly assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      nstep_between_samples, setup.get_simulation_type(),
      setup.instantiate_property_reader());
  time_scheme->link_assembly(assembly);

  benchmark<true>(assembly, time_scheme);
  benchmark<false>(assembly, time_scheme);
  benchmark<true>(assembly, time_scheme);
  benchmark<false>(assembly, time_scheme);
  std::cout << std::endl;
}

} // namespace benchmarks
} // namespace specfem

int main(int argc, char **argv) {
  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  {
    const std::string default_file = __default_file__;
    const YAML::Node default_dict = YAML::LoadFile(default_file);
    std::cout << "Acoustic:" << std::endl;
    specfem::benchmarks::run_benchmark(YAML::LoadFile(__benchmark_iso__),
                                       default_dict, mpi);
    std::cout << "Elastic isotropic:" << std::endl;
    specfem::benchmarks::run_benchmark(YAML::LoadFile(__benchmark_eiso__),
                                       default_dict, mpi);
    std::cout << "Elastic anisotropic:" << std::endl;
    specfem::benchmarks::run_benchmark(YAML::LoadFile(__benchmark_eani__),
                                       default_dict, mpi);
  }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
