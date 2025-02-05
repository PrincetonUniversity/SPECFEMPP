#include "execute.hpp"
#include "stiffness.hpp"
// #include "divide.hpp"

namespace specfem {
namespace benchmarks {

void benchmark(specfem::compute::assembly &assembly,
               std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
               const int &flag) {
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto isotropic = specfem::element::property_tag::isotropic;
  constexpr auto anisotropic = specfem::element::property_tag::anisotropic;

  const int nstep = time_scheme->get_max_timestep();

  const auto solver_start_time = std::chrono::system_clock::now();

  for (const auto [istep, dt] : time_scheme->iterate_forward()) {
    compute_stiffness_interaction<acoustic, isotropic>(assembly, istep, flag);
    compute_stiffness_interaction<elastic, isotropic>(assembly, istep, flag);
    compute_stiffness_interaction<elastic, anisotropic>(assembly, istep, flag);
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

  benchmark(assembly, time_scheme, 1);
  benchmark(assembly, time_scheme, 0);
  // benchmark(assembly, time_scheme, 1);
  // benchmark(assembly, time_scheme, 0);
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
