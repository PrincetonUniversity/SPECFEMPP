#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "execute.hpp"
#include "kokkos_kernels/impl/compute_mass_matrix.hpp"
#include "kokkos_kernels/impl/compute_seismogram.hpp"
#include "kokkos_kernels/impl/compute_source_interaction.hpp"
#include "kokkos_kernels/impl/compute_stiffness_interaction.hpp"
#include "kokkos_kernels/impl/divide_mass_matrix.hpp"
#include "kokkos_kernels/impl/interface_kernels.hpp"
#include "kokkos_kernels/impl/invert_mass_matrix.hpp"

namespace specfem {
namespace kokkos_kernels {
template <specfem::element::medium_tag medium>
inline void update_wavefields(specfem::compute::assembly &assembly,
                              const int istep) {
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto wavefield =
      specfem::wavefield::simulation_field::forward;
  constexpr static auto ngll = 5;

#define CALL_STIFFNESS_FORCE_UPDATE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,   \
                                    BOUNDARY_TAG)                              \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    impl::compute_stiffness_interaction<                                       \
        dimension, wavefield, ngll, GET_TAG(MEDIUM_TAG),                       \
        GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(assembly, istep);        \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      CALL_STIFFNESS_FORCE_UPDATE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
              WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_NONE,
                    BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                    BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef CALL_STIFFNESS_FORCE_UPDATE

#define CALL_DIVIDE_MASS_MATRIX_FUNCTION(DIMENSION_TAG, MEDIUM_TAG)            \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    impl::divide_mass_matrix<dimension, wavefield, GET_TAG(MEDIUM_TAG)>(       \
        assembly);                                                             \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      CALL_DIVIDE_MASS_MATRIX_FUNCTION,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef CALL_DIVIDE_MASS_MATRIX_FUNCTION
}

} // namespace kokkos_kernels
} // namespace specfem

void benchmark(specfem::compute::assembly &assembly,
               std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme) {
  constexpr auto elastic = specfem::element::medium_tag::elastic;

  const int nstep = time_scheme->get_max_timestep();

  for (const auto [istep, dt] : time_scheme->iterate_forward()) {
    time_scheme->apply_predictor_phase_forward(elastic);

    specfem::kokkos_kernels::update_wavefields<elastic>(assembly, istep);
    time_scheme->apply_corrector_phase_forward(elastic);

    if ((istep + 1) % 400 == 0) {
      std::cout << "Progress : executed " << istep + 1 << " steps of " << nstep
                << " steps" << std::endl;
    }
  }

  std::cout << std::endl;
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
  mpi->cout("-------------------------------");
  const std::vector<std::shared_ptr<specfem::sources::source> > sources;
  const std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers;
  const int nsteps = setup.get_nsteps();
  specfem::compute::assembly assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      nstep_between_samples, setup.get_simulation_type(),
      setup.instantiate_property_reader());
  time_scheme->link_assembly(assembly);

  const auto solver_start_time = std::chrono::system_clock::now();
  benchmark(assembly, time_scheme);
  const auto solver_end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> solver_time =
      solver_end_time - solver_start_time;
  std::cout << "Solver time: " << solver_time.count() << "s\n" << std::endl;
}

int main(int argc, char **argv) {
  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  {
    const std::string default_file = __default_file__;
    const YAML::Node default_dict = YAML::LoadFile(default_file);
    std::cout << "Elastic isotropic" << std::endl;
    run_benchmark(YAML::LoadFile(__benchmark_iso__), default_dict, mpi);
    std::cout << "Elastic anisotropic" << std::endl;
    run_benchmark(YAML::LoadFile(__benchmark_aniso__), default_dict, mpi);
  }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
