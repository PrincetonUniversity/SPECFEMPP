#include "specfem/solver/implicit_solver.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/compute.tpp"
#include "specfem/compute/impl/compute_source_interaction.hpp"
#include "specfem/linear_system/damping_assembler.hpp"
#include "specfem/linear_system/mass_vector.hpp"
#include "specfem/linear_system/sparse_matrix_view/field_vector.hpp"
#include "specfem/linear_system/sparse_matrix_view/matrix_view.hpp"
#include "specfem/linear_system/tpetra_assembler.hpp"
#include "specfem/linear_system/vector_view/vector_view.hpp"
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
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
specfem::solver::ImplicitNewmarkSolver<Tags>::ImplicitNewmarkSolver(
    const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
    const std::vector<
        std::shared_ptr<specfem::periodic_tasks::periodic_task<dimension_tag>>>
        &tasks,
    AssemblyType assembly, const ImplicitSolverConfig &config)
    : time_scheme_(time_scheme), tasks_(tasks), assembly_(assembly),
      config_(config) {

  if (config_.newmark.beta < 0) {
    throw std::runtime_error(
        "specfem::solver::ImplicitNewmarkSolver: Newmark beta must be "
        "non-negative.");
  }

  if (config_.newmark.gamma < 0 || config_.newmark.gamma > 1) {
    throw std::runtime_error(
        "specfem::solver::ImplicitNewmarkSolver: Newmark gamma must lie in "
        "[0, 1].");
  }

  // Reject out-of-scope meshes before building the dof maps and graphs: the
  // element-dense stiffness graph costs gigabytes on a moderately sized mesh,
  // and the assemblers below would only discard it and throw.
  specfem::linear_system::validate_single_medium<Tags>(assembly_);
  specfem::linear_system::validate_stiffness_scope<Tags>(
      assembly_, specfem::linear_system::StiffnessScope::with_stacey);

  // One description of the mesh -- dof maps, the element-dense stiffness
  // graph, the block-diagonal damping graph -- shared by every operator
  // assembled below. Each graph costs two host passes over the connectivity,
  // so letting the assemblers each build their own would pay for them twice.
  fe_ = std::make_unique<FEAssemblyType>(MappingType(assembly_));

  specfem::linear_system::StiffnessAssembler<Tags> stiffness_assembler(
      assembly_, *fe_, specfem::linear_system::StiffnessScope::with_stacey);
  stiffness_ = stiffness_assembler.assemble();

  specfem::linear_system::DampingAssembler<Tags> damping_assembler(assembly_,
                                                                   *fe_);
  damping_ = damping_assembler.assemble();

  mass_ = specfem::linear_system::assemble_mass_vector<Tags>(assembly_, *fe_);

  vectors_ =
      std::make_unique<specfem::linear_system::VectorSpace>(fe_->owned_map());
  u_.emplace(vectors_->vector());
  v_.emplace(vectors_->vector());
  a_.emplace(vectors_->vector());
  u_new_.emplace(vectors_->vector());
  v_new_.emplace(vectors_->vector());
  a_new_.emplace(vectors_->vector());
  rhs_.emplace(vectors_->vector());

  form_operator(time_scheme_->get_timestep());
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::solver::ImplicitNewmarkSolver<Tags>::form_operator(
    const type_real dt) {
  using scalar_type = specfem::linear_system::scalar_type;
  using crs_matrix_type = specfem::linear_system::crs_matrix_type;

  const type_real beta = config_.newmark.beta;
  const type_real gamma = config_.newmark.gamma;

  const scalar_type stiffness_coefficient =
      static_cast<scalar_type>(beta * dt * dt);
  const scalar_type damping_coefficient = static_cast<scalar_type>(gamma * dt);

  // ADL cannot reach an operator in specfem::linear_system when both operands
  // are a scalar and a Tpetra type, so the scaled-matrix spelling below needs
  // this declaration. diag() returns one of our own types and needs none.
  using specfem::linear_system::operator*;

  // A lives on K's graph: every C entry is a same-point pair, which is a
  // same-element pair, and every M entry is a self-pair -- so both additions
  // are contained in it. The view checks that per row rather than trusting it.
  specfem::linear_system::SparseMatrixView<MappingType> system(
      fe_->full_matrix_graph(), fe_->mapping());

  system.begin_fill();
  system += stiffness_coefficient * stiffness();  // beta dt^2 K
  system += damping_coefficient * damping();      // + gamma dt C
  system += specfem::linear_system::diag(mass()); // + M
  system.finalize();

  system_operator_ = system.matrix();

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
          system_operator_, a_new_->rcp(), rhs_->rcp()));
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
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::solver::ImplicitNewmarkSolver<Tags>::extract_source_vector(
    const int istep, specfem::linear_system::VectorView &f) {
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

  specfem::linear_system::copy_field_to_vector(fe_->mapping(),
                                               host_acceleration, f.vector());
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::solver::ImplicitNewmarkSolver<Tags>::write_state_to_fields() {
  constexpr auto forward = specfem::simulation::field_type::forward;

  auto &field = assembly_.fields.template get_simulation_field<forward>();
  const auto &field_impl = field.template get_field<medium_tag>();
  const auto h_u = field_impl.get_host_field();
  const auto h_v = field_impl.get_host_field_dot();
  const auto h_a = field_impl.get_host_field_dot_dot();

  const auto &mapping = fe_->mapping();
  specfem::linear_system::copy_vector_to_field(mapping, u_->vector(), h_u);
  specfem::linear_system::copy_vector_to_field(mapping, v_->vector(), h_v);
  specfem::linear_system::copy_vector_to_field(mapping, a_->vector(), h_a);

  // Copy only the three state views (not fields.copy_to_device(), which
  // would also touch the mass storage the assemblers treat as scratch).
  Kokkos::deep_copy(field_impl.get_field(), h_u);
  Kokkos::deep_copy(field_impl.get_field_dot(), h_v);
  Kokkos::deep_copy(field_impl.get_field_dot_dot(), h_a);
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::solver::ImplicitNewmarkSolver<Tags>::apply_predictor_phase() {
  using scalar_type = specfem::linear_system::scalar_type;

  const type_real dt = time_scheme_->get_timestep();
  const scalar_type displacement_coefficient = static_cast<scalar_type>(
      dt * dt * (static_cast<type_real>(0.5) - config_.newmark.beta));
  const scalar_type velocity_coefficient =
      static_cast<scalar_type>(dt * (1 - config_.newmark.gamma));

  const auto &u = *u_;
  const auto &v = *v_;
  const auto &a = *a_;
  // The predictors live in u_new_ / v_new_ until the corrector updates them.
  auto &u_pred = *u_new_;
  auto &v_pred = *v_new_;

  u_pred = u + dt * v + displacement_coefficient * a;
  v_pred = v + velocity_coefficient * a;
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::solver::ImplicitNewmarkSolver<Tags>::compute_forces(
    const int istep) {
  const auto &u_pred = *u_new_;
  const auto &v_pred = *v_new_;
  auto &b = *rhs_;

  // f_{n+1}: the explicit loop pairs STF(istep = n) with the state at
  // t_{n+1}; the implicit loop must match.
  extract_source_vector(istep, b);
  b -= stiffness() * u_pred;
  if (damping().getGlobalNumEntries() > 0) {
    b -= damping() * v_pred;
  }
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::solver::ImplicitNewmarkSolver<Tags>::solve(const int istep) {
  const auto &b = *rhs_;
  auto &a_new = *a_new_;

  a_new = *a_;
  problem_->setProblem(a_new.rcp(), b.rcp());
  if (gmres_->solve() == Belos::Converged) {
    return;
  }

  // Single-precision PseudoBlockGmres can abort with a "loss of accuracy"
  // flag while the solution is fine; trust only the true residual b - A x.
  const type_real residual_norm =
      specfem::linear_system::norm2(b - system_operator() * a_new);
  const type_real rhs_norm = specfem::linear_system::norm2(b);
  if (!(residual_norm <= config_.gmres_tolerance * rhs_norm)) {
    std::ostringstream message;
    message << "specfem::solver::ImplicitNewmarkSolver: GMRES did not "
               "converge at step "
            << istep << " (" << gmres_->getNumIters()
            << " iterations, true relative residual "
            << residual_norm / rhs_norm << ", requested "
            << config_.gmres_tolerance << ").";
    throw std::runtime_error(message.str());
  }
  std::ostringstream message;
  message << "GMRES flagged loss of accuracy at step " << istep
          << " but the true relative residual " << residual_norm / rhs_norm
          << " meets the tolerance; "
          << "continuing.";
  specfem::Logger::info(message.str());
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::solver::ImplicitNewmarkSolver<Tags>::apply_corrector_phase() {
  using scalar_type = specfem::linear_system::scalar_type;

  const type_real dt = time_scheme_->get_timestep();
  const scalar_type displacement_coefficient =
      static_cast<scalar_type>(config_.newmark.beta * dt * dt);
  const scalar_type velocity_coefficient =
      static_cast<scalar_type>(config_.newmark.gamma * dt);

  const auto &a_new = *a_new_;
  auto &u_new = *u_new_;
  auto &v_new = *v_new_;

  u_new += displacement_coefficient * a_new;
  v_new += velocity_coefficient * a_new;
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::solver::ImplicitNewmarkSolver<Tags>::run() {
  constexpr auto forward = specfem::simulation::field_type::forward;

  const int nstep = time_scheme_->get_max_timestep();

  // Named locally so that the updates below read as the equations they are.
  auto &u = *u_;
  auto &v = *v_;
  auto &a = *a_;
  auto &u_new = *u_new_;
  auto &v_new = *v_new_;
  auto &a_new = *a_new_;

  u = 0;
  v = 0;
  a = 0;
  last_step_ = 0;
  // The sample counter lives on the time scheme and would otherwise carry
  // over, so a re-run would write past the end of the seismogram buffer.
  time_scheme_->reset_seismogram_step();

  const bool check_steady_state = config_.steady_state_tolerance > 0;
  type_real velocity_scale = 0;
  type_real acceleration_scale = 0;

  for (const auto &task : tasks_) {
    task->initialize(assembly_);
  }

  for (const auto [istep, dt_step] : time_scheme_->iterate_forward()) {
    (void)dt_step;

    apply_predictor_phase();
    compute_forces(istep);
    solve(istep);
    apply_corrector_phase();

    bool steady = false;
    if (check_steady_state) {
      const type_real velocity_increment =
          specfem::linear_system::norm2(v_new - v);
      const type_real acceleration_increment =
          specfem::linear_system::norm2(a_new - a);
      // Increments relative to the running maxima -- the natural problem
      // scales; a displacement criterion would never fire on the
      // constant-velocity drift asymptote (see ImplicitSolverConfig).
      velocity_scale =
          std::max(velocity_scale, specfem::linear_system::norm2(v_new));
      acceleration_scale =
          std::max(acceleration_scale, specfem::linear_system::norm2(a_new));
      steady = velocity_scale > 0 && acceleration_scale > 0 &&
               velocity_increment <=
                   config_.steady_state_tolerance * velocity_scale &&
               acceleration_increment <=
                   config_.steady_state_tolerance * acceleration_scale;
      std::ostringstream message;
      message << "Steady-state check at step " << istep << ": |dv|/max|v| = "
              << (velocity_scale > 0 ? velocity_increment / velocity_scale : 0)
              << ", |da|/max|a| = "
              << (acceleration_scale > 0
                      ? acceleration_increment / acceleration_scale
                      : 0)
              << " (tolerance " << config_.steady_state_tolerance << ")";
      specfem::Logger::info(message.str());
    }

    swap(u, u_new);
    swap(v, v_new);
    swap(a, a_new);

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
