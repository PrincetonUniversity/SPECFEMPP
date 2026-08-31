#include "specfem/solver/implicit_solver.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/compute.tpp"
#include "specfem/compute/impl/compute_source_interaction.hpp"
#include "specfem/linear_system/damping_assembler.hpp"
#include "specfem/linear_system/mass_vector.hpp"
#include "specfem/linear_system/tpetra_assembler.hpp"
#include "specfem/logger.hpp"
#include "specfem/tags.hpp"
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Ifpack2_Factory.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

template <typename Tags>
specfem::solver::ImplicitNewmarkSolver<Tags>::ImplicitNewmarkSolver(
    const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
    const std::vector<
        std::shared_ptr<specfem::periodic_tasks::periodic_task<dimension_tag>>>
        &tasks,
    AssemblyType assembly, const ImplicitSolverConfig &config)
    : time_scheme_(time_scheme), tasks_(tasks), assembly_(assembly),
      config_(config) {

  if (!(config_.newmark.beta > 0)) {
    throw std::runtime_error(
        "specfem::solver::ImplicitNewmarkSolver: Newmark beta must be > 0 "
        "(the explicit beta = 0 limit has no displacement-form operator); "
        "use the explicit time_marching solver instead.");
  }

  specfem::linear_system::StiffnessAssembler<Tags> stiffness_assembler(
      assembly_,
      specfem::linear_system::StiffnessAssembler<Tags>::default_batch_size,
      specfem::linear_system::StiffnessScope::with_stacey);
  stiffness_ = stiffness_assembler.assemble();
  dof_map_ = std::make_unique<specfem::linear_system::DofMap>(
      stiffness_assembler.dof_map());

  specfem::linear_system::DampingAssembler<Tags> damping_assembler(assembly_,
                                                                   *dof_map_);
  damping_ = damping_assembler.assemble();

  mass_ =
      specfem::linear_system::assemble_mass_vector<Tags>(assembly_, *dof_map_);

  const auto map = dof_map_->owned_map();
  u_ = Teuchos::rcp(new specfem::linear_system::vector_type(map));
  v_ = Teuchos::rcp(new specfem::linear_system::vector_type(map));
  a_ = Teuchos::rcp(new specfem::linear_system::vector_type(map));
  u_new_ = Teuchos::rcp(new specfem::linear_system::vector_type(map));
  v_new_ = Teuchos::rcp(new specfem::linear_system::vector_type(map));
  a_new_ = Teuchos::rcp(new specfem::linear_system::vector_type(map));
  rhs_ = Teuchos::rcp(new specfem::linear_system::vector_type(map));
  tmp_ = Teuchos::rcp(new specfem::linear_system::vector_type(map));
  tmp2_ = Teuchos::rcp(new specfem::linear_system::vector_type(map));

  form_operator(time_scheme_->get_timestep());
}

template <typename Tags>
void specfem::solver::ImplicitNewmarkSolver<Tags>::form_operator(
    const type_real dt) {
  using scalar_type = specfem::linear_system::scalar_type;
  using crs_matrix_type = specfem::linear_system::crs_matrix_type;
  using global_ordinal_type = specfem::linear_system::global_ordinal_type;

  const type_real beta = config_.newmark.beta;
  const type_real gamma = config_.newmark.gamma;
  const scalar_type mass_coefficient =
      static_cast<scalar_type>(1) / (beta * dt * dt);
  const scalar_type damping_coefficient =
      static_cast<scalar_type>(gamma / (beta * dt));

  // A on K's static graph: every C block entry is a same-element pair and
  // the M diagonal is a self-pair, so both sumInto below always hit.
  auto system = Teuchos::rcp(new crs_matrix_type(stiffness_->getCrsGraph()));

  const global_ordinal_type num_rows = dof_map_->num_global_dofs();
  typename crs_matrix_type::nonconst_global_inds_host_view_type columns(
      "specfem::solver::implicit_operator_columns",
      stiffness_->getGlobalMaxNumRowEntries());
  typename crs_matrix_type::nonconst_values_host_view_type values(
      "specfem::solver::implicit_operator_values",
      stiffness_->getGlobalMaxNumRowEntries());

  for (global_ordinal_type row = 0; row < num_rows; ++row) {
    std::size_t row_entries = 0;
    stiffness_->getGlobalRowCopy(row, columns, values, row_entries);
    const int replaced = system->replaceGlobalValues(
        row, static_cast<int>(row_entries), values.data(), columns.data());
    if (replaced != static_cast<int>(row_entries)) {
      throw std::runtime_error(
          "specfem::solver::ImplicitNewmarkSolver: copying K into the "
          "system operator failed; the graphs disagree.");
    }
  }

  if (damping_->getGlobalNumEntries() > 0) {
    typename crs_matrix_type::nonconst_global_inds_host_view_type
        damping_columns("specfem::solver::implicit_damping_columns",
                        damping_->getGlobalMaxNumRowEntries());
    typename crs_matrix_type::nonconst_values_host_view_type damping_values(
        "specfem::solver::implicit_damping_values",
        damping_->getGlobalMaxNumRowEntries());
    for (global_ordinal_type row = 0; row < num_rows; ++row) {
      std::size_t row_entries = 0;
      damping_->getGlobalRowCopy(row, damping_columns, damping_values,
                                 row_entries);
      if (row_entries == 0) {
        continue;
      }
      for (std::size_t k = 0; k < row_entries; ++k) {
        damping_values(k) *= damping_coefficient;
      }
      const int updated = system->sumIntoGlobalValues(
          row, static_cast<int>(row_entries), damping_values.data(),
          damping_columns.data());
      if (updated != static_cast<int>(row_entries)) {
        throw std::runtime_error(
            "specfem::solver::ImplicitNewmarkSolver: summing C into the "
            "system operator failed; a damping entry is outside K's graph.");
      }
    }
  }

  {
    const auto mass_view = mass_->getLocalViewHost(Tpetra::Access::ReadOnly);
    for (global_ordinal_type row = 0; row < num_rows; ++row) {
      const scalar_type diagonal_term =
          mass_view(static_cast<std::size_t>(row), 0) * mass_coefficient;
      const int updated =
          system->sumIntoGlobalValues(row, 1, &diagonal_term, &row);
      if (updated != 1) {
        throw std::runtime_error(
            "specfem::solver::ImplicitNewmarkSolver: the system operator's "
            "graph is missing a diagonal entry.");
      }
    }
  }

  system->fillComplete(dof_map_->owned_map(), dof_map_->owned_map());
  system_operator_ = system;

  // MueLu (AMG) deferred: the float-only TROMP Trilinos installs do not
  // link it (MueLu references Xpetra::Matrix<double> unconditionally);
  // revisit with the follow-up of issue #1984.
  using row_matrix_type =
      Tpetra::RowMatrix<scalar_type, crs_matrix_type::local_ordinal_type,
                        crs_matrix_type::global_ordinal_type,
                        crs_matrix_type::node_type>;
  auto preconditioner = Ifpack2::Factory::create<row_matrix_type>(
      config_.preconditioner, system_operator_);
  Teuchos::ParameterList preconditioner_params = config_.preconditioner_params;
  if (config_.preconditioner == "RILUK" &&
      !preconditioner_params.isParameter("fact: iluk level-of-fill")) {
    // ILU(0): the fill pattern is A's own graph. Spectral-element rows carry
    // ~2000 nonzeros (all dofs of all adjacent elements), so level-1 fill
    // (~the pattern of A^2) explodes combinatorially -- RILUK's symbolic
    // setup runs for minutes and gigabytes. Override via
    // preconditioner_params for small problems.
    preconditioner_params.set("fact: iluk level-of-fill", 0);
  }
  preconditioner->setParameters(preconditioner_params);
  preconditioner->initialize();
  preconditioner->compute();
  preconditioner_ = preconditioner;

  problem_ = Teuchos::rcp(
      new Belos::LinearProblem<scalar_type, multivector_type, operator_type>(
          system_operator_, u_new_, rhs_));
  // Right preconditioning keeps the convergence test on the true residual.
  problem_->setRightPrec(preconditioner_);

  auto belos_params = Teuchos::rcp(new Teuchos::ParameterList());
  belos_params->set("Convergence Tolerance", config_.gmres_tolerance);
  belos_params->set("Maximum Iterations", config_.gmres_max_iterations);
  belos_params->set("Num Blocks", config_.gmres_restart_length);
  belos_params->set("Verbosity", Belos::Errors + Belos::Warnings);
  gmres_ = Teuchos::rcp(
      new Belos::PseudoBlockGmresSolMgr<scalar_type, multivector_type,
                                        operator_type>(problem_, belos_params));
}

template <typename Tags>
void specfem::solver::ImplicitNewmarkSolver<Tags>::extract_source_vector(
    const int istep, specfem::linear_system::vector_type &f) {
  constexpr auto forward = specfem::simulation::field_type::forward;

  auto &field = assembly_.fields.template get_simulation_field<forward>();
  const auto &field_impl = field.template get_field<medium_tag>();
  const auto device_acceleration = field_impl.get_field_dot_dot();
  const auto host_acceleration = field_impl.get_host_field_dot_dot();

  // The source kernel only adds into the acceleration field, and the state
  // is rewritten by write_state_to_fields() at the end of the step, so
  // zeroing here is the only bookkeeping the probe needs.
  Kokkos::deep_copy(device_acceleration, 0);
  specfem::compute::impl::compute_source_interaction<
      5, specfem::tags::Tags<dimension_tag, forward, medium_tag>>(assembly_,
                                                                  istep);
  Kokkos::deep_copy(host_acceleration, device_acceleration);

  auto view = f.getLocalViewHost(Tpetra::Access::OverwriteAll);
  for (int iglob = 0; iglob < dof_map_->nglob(); ++iglob) {
    for (int icomp = 0; icomp < dof_map_->ncomp(); ++icomp) {
      view(static_cast<std::size_t>(dof_map_->gid(iglob, icomp)), 0) =
          host_acceleration(iglob, icomp);
    }
  }
}

template <typename Tags>
void specfem::solver::ImplicitNewmarkSolver<Tags>::write_state_to_fields() {
  constexpr auto forward = specfem::simulation::field_type::forward;

  auto &field = assembly_.fields.template get_simulation_field<forward>();
  const auto &field_impl = field.template get_field<medium_tag>();
  const auto h_u = field_impl.get_host_field();
  const auto h_v = field_impl.get_host_field_dot();
  const auto h_a = field_impl.get_host_field_dot_dot();

  {
    const auto u_view = u_->getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto v_view = v_->getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto a_view = a_->getLocalViewHost(Tpetra::Access::ReadOnly);
    for (int iglob = 0; iglob < dof_map_->nglob(); ++iglob) {
      for (int icomp = 0; icomp < dof_map_->ncomp(); ++icomp) {
        const auto dof = static_cast<std::size_t>(dof_map_->gid(iglob, icomp));
        h_u(iglob, icomp) = u_view(dof, 0);
        h_v(iglob, icomp) = v_view(dof, 0);
        h_a(iglob, icomp) = a_view(dof, 0);
      }
    }
  }

  // Copy only the three state views (not fields.copy_to_device(), which
  // would also touch the mass storage the assemblers treat as scratch).
  Kokkos::deep_copy(field_impl.get_field(), h_u);
  Kokkos::deep_copy(field_impl.get_field_dot(), h_v);
  Kokkos::deep_copy(field_impl.get_field_dot_dot(), h_a);
}

template <typename Tags>
void specfem::solver::ImplicitNewmarkSolver<Tags>::run() {
  constexpr auto forward = specfem::simulation::field_type::forward;
  using scalar_type = specfem::linear_system::scalar_type;

  const type_real dt = time_scheme_->get_timestep();
  const type_real beta = config_.newmark.beta;
  const type_real gamma = config_.newmark.gamma;
  const int nstep = time_scheme_->get_max_timestep();

  // Displacement-form Newmark coefficients (see the class docs).
  const scalar_type c_a0 = static_cast<scalar_type>(1 / (beta * dt * dt));
  const scalar_type c_a1 = static_cast<scalar_type>(1 / (beta * dt));
  const scalar_type c_a2 = static_cast<scalar_type>(1 / (2 * beta) - 1);
  const scalar_type c_c0 = static_cast<scalar_type>(gamma / (beta * dt));
  const scalar_type c_c1 = static_cast<scalar_type>(gamma / beta - 1);
  const scalar_type c_c2 =
      static_cast<scalar_type>(dt * (gamma / (2 * beta) - 1));
  const scalar_type c_v0 = static_cast<scalar_type>(dt * (1 - gamma));
  const scalar_type c_v1 = static_cast<scalar_type>(dt * gamma);

  u_->putScalar(0);
  v_->putScalar(0);
  a_->putScalar(0);
  last_step_ = 0;

  const bool has_damping = damping_->getGlobalNumEntries() > 0;
  const bool check_steady_state = config_.steady_state_tolerance > 0;
  type_real velocity_scale = 0;
  type_real acceleration_scale = 0;

  for (const auto &task : tasks_) {
    task->initialize(assembly_);
  }

  for (const auto [istep, dt_step] : time_scheme_->iterate_forward()) {
    (void)dt_step;

    // b = f_{n+1}: the explicit loop pairs STF(istep = n) with the state at
    // t_{n+1}; the implicit loop must match.
    extract_source_vector(istep, *rhs_);

    if (has_damping) {
      tmp_->update(c_c0, *u_, c_c1, *v_, 0);
      tmp_->update(c_c2, *a_, static_cast<scalar_type>(1));
      damping_->apply(*tmp_, *tmp2_);
      rhs_->update(static_cast<scalar_type>(1), *tmp2_,
                   static_cast<scalar_type>(1));
    }

    tmp_->update(c_a0, *u_, c_a1, *v_, 0);
    tmp_->update(c_a2, *a_, static_cast<scalar_type>(1));
    rhs_->elementWiseMultiply(static_cast<scalar_type>(1), *mass_, *tmp_,
                              static_cast<scalar_type>(1));

    // Warm start from u_n: near steady state GMRES converges in a few
    // iterations.
    u_new_->update(static_cast<scalar_type>(1), *u_, 0);
    problem_->setProblem(u_new_, rhs_);
    if (gmres_->solve() != Belos::Converged) {
      std::ostringstream message;
      message << "specfem::solver::ImplicitNewmarkSolver: GMRES did not "
                 "converge at step "
              << istep << " (" << gmres_->getNumIters()
              << " iterations, achieved tolerance " << gmres_->achievedTol()
              << ", requested " << config_.gmres_tolerance << ").";
      throw std::runtime_error(message.str());
    }

    // a_{n+1} = c_a0 (u_{n+1} - u_n - dt v_n) - c_a2 a_n
    a_new_->update(c_a0, *u_new_, -c_a0, *u_, 0);
    a_new_->update(static_cast<scalar_type>(-c_a0 * dt), *v_,
                   static_cast<scalar_type>(1));
    a_new_->update(-c_a2, *a_, static_cast<scalar_type>(1));
    // v_{n+1} = v_n + dt (1 - gamma) a_n + dt gamma a_{n+1}
    v_new_->update(static_cast<scalar_type>(1), *v_, c_v0, *a_, 0);
    v_new_->update(c_v1, *a_new_, static_cast<scalar_type>(1));

    bool steady = false;
    if (check_steady_state) {
      tmp_->update(static_cast<scalar_type>(1), *v_new_,
                   static_cast<scalar_type>(-1), *v_, 0);
      const type_real velocity_increment = tmp_->norm2();
      tmp_->update(static_cast<scalar_type>(1), *a_new_,
                   static_cast<scalar_type>(-1), *a_, 0);
      const type_real acceleration_increment = tmp_->norm2();
      // Increments relative to the running maxima -- the natural problem
      // scales; a displacement criterion would never fire on the
      // constant-velocity drift asymptote (see ImplicitSolverConfig).
      velocity_scale =
          std::max(velocity_scale, static_cast<type_real>(v_new_->norm2()));
      acceleration_scale =
          std::max(acceleration_scale, static_cast<type_real>(a_new_->norm2()));
      steady = velocity_scale > 0 && acceleration_scale > 0 &&
               velocity_increment <=
                   config_.steady_state_tolerance * velocity_scale &&
               acceleration_increment <=
                   config_.steady_state_tolerance * acceleration_scale;
    }

    std::swap(u_, u_new_);
    std::swap(v_, v_new_);
    std::swap(a_, a_new_);

    write_state_to_fields();

    if (time_scheme_->compute_seismogram(istep)) {
      specfem::compute::compute_seismograms<
          5, specfem::tags::Tags<dimension_tag, forward>>(
          assembly_, time_scheme_->get_seismogram_step());
      time_scheme_->increment_seismogram_step();
    }
    for (const auto &task : tasks_) {
      if (task && task->should_run(istep + 1)) {
        task->run(assembly_, istep + 1);
      }
    }

    last_step_ = istep + 1;

    if (istep % 10 == 0) {
      std::ostringstream message;
      message << "Progress : executed " << istep << " steps of " << nstep
              << " steps (GMRES: " << gmres_->getNumIters() << " iterations)"
              << std::endl;
      specfem::Logger::info(message.str());
    }

    if (steady) {
      std::ostringstream message;
      message << "Steady state reached at step " << last_step_ << " of "
              << nstep << " (velocity and acceleration increments below "
              << config_.steady_state_tolerance << ").";
      specfem::Logger::info(message.str());
      break;
    }
  }

  for (const auto &task : tasks_) {
    if (task && !task->should_run(last_step_) && task->should_run(-1)) {
      task->run(assembly_, last_step_);
    }
  }

  for (const auto &task : tasks_) {
    task->finalize(assembly_);
  }

  specfem::Logger::info(" -- Implicit simulation complete. -- \n");
}

namespace specfem::solver_impl {
/// Tag bundle for the only combination explicitly instantiated for the
/// implicit solver (issue #1984), matching the linear_system scope.
using elastic_isotropic_tags =
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
} // namespace specfem::solver_impl

// Explicit instantiation: 3D elastic isotropic
template class specfem::solver::ImplicitNewmarkSolver<
    specfem::solver_impl::elastic_isotropic_tags>;

#endif // SPECFEM_ENABLE_TRILINOS
