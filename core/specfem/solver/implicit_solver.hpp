#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "solver.hpp"
#include "specfem/enums.hpp"
#include "specfem/linear_system/sparse_matrix_view/fe_assembly.hpp"
#include "specfem/periodic_tasks.hpp"
#include "specfem/timescheme.hpp"
#include <BelosLinearProblem.hpp>
#include <BelosSolverManager.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <memory>
#include <string>
#include <vector>

namespace specfem {
namespace solver {

/**
 * @brief Newmark-beta parameters of the implicit update.
 *
 * The defaults (\f$ \beta = 1/4, \gamma = 1/2 \f$, average acceleration) are
 * unconditionally stable and non-dissipative. For steady-state driving use
 * @ref dissipative: \f$ \gamma > 1/2 \f$ introduces algorithmic damping of
 * the modes a large time step cannot resolve, and
 * \f$ \beta = (\gamma + 1/2)^2 / 4 \f$ keeps the scheme unconditionally
 * stable and second-order in the resolved modes.
 */
struct NewmarkBetaParameters {
  type_real beta = static_cast<type_real>(0.25); ///< Newmark \f$ \beta > 0 \f$
  type_real gamma = static_cast<type_real>(0.5); ///< Newmark \f$ \gamma \f$

  /**
   * @brief Dissipative preset for driving a run to steady state.
   *
   * @param gamma Newmark \f$ \gamma > 1/2 \f$; larger means stronger
   *        high-frequency dissipation
   * @return Parameters with \f$ \beta = (\gamma + 1/2)^2 / 4 \f$
   */
  static NewmarkBetaParameters dissipative(const type_real gamma) {
    NewmarkBetaParameters parameters;
    parameters.gamma = gamma;
    parameters.beta = (gamma + static_cast<type_real>(0.5)) *
                      (gamma + static_cast<type_real>(0.5)) /
                      static_cast<type_real>(4);
    return parameters;
  }
};

/**
 * @brief Linear-solver and stopping configuration of the implicit solver.
 */
struct ImplicitSolverConfig {
  NewmarkBetaParameters newmark{}; ///< Newmark update parameters

  /// GMRES relative residual tolerance. Floor ~1e-6 in single precision.
  specfem::linear_system::scalar_type gmres_tolerance =
      static_cast<specfem::linear_system::scalar_type>(1e-5);
  int gmres_max_iterations = 1000; ///< GMRES iteration cap per solve
  int gmres_restart_length = 100;  ///< GMRES restart (Belos "Num Blocks")

  /// Ifpack2 factory name: "RILUK" (default) or "RELAXATION" (symmetric
  /// Gauss-Seidel fallback if RILUK misbehaves on a platform).
  std::string preconditioner = "RILUK";
  Teuchos::ParameterList preconditioner_params{}; ///< Ifpack2 parameters

  /**
   * @brief Early-stop tolerance for steady-state driving; 0 disables (run
   * all steps).
   *
   * The run stops once the velocity and acceleration increments both drop
   * below the tolerance relative to their running maxima:
   * \f$ \| v_{n+1} - v_n \| \le \mathrm{tol} \cdot \max_k \| v_k \| \f$ and
   * the same for \f$ a \f$. Velocity-based on purpose: a nonzero-net-force
   * source on a Stacey-truncated box converges to a constant-velocity drift
   * superposed on the converged deformation, so a displacement increment
   * never vanishes.
   */
  type_real steady_state_tolerance = 0;
};

/**
 * @brief Implicit Newmark solver: one Belos GMRES solve per time step on the
 * assembled operator \f$ A = M / (\beta \Delta t^2) + \gamma / (\beta \Delta
 * t) \, C + K \f$.
 *
 * Solves the semi-discrete equation of motion
 * \f$ M \ddot{u} + C \dot{u} + K u = f \f$ with the Newmark-beta update
 * \f[ u_{n+1} = u_n + \Delta t \, v_n + \Delta t^2 \left[ (1/2 - \beta) a_n
 * + \beta \, a_{n+1} \right], \quad
 * v_{n+1} = v_n + \Delta t \left[ (1 - \gamma) a_n + \gamma \, a_{n+1}
 * \right] \f]
 * in displacement form: \f$ A u_{n+1} = b \f$ with
 * \f[ b = f_{n+1}
 * + M \circ \left[ \frac{u_n}{\beta \Delta t^2} + \frac{v_n}{\beta \Delta t}
 *   + \left( \frac{1}{2\beta} - 1 \right) a_n \right]
 * + C \left[ \frac{\gamma}{\beta \Delta t} u_n
 *   + \left( \frac{\gamma}{\beta} - 1 \right) v_n
 *   + \Delta t \left( \frac{\gamma}{2\beta} - 1 \right) a_n \right]. \f]
 * Acceleration and velocity then follow as
 * \f$ a_{n+1} = (u_{n+1} - u_n - \Delta t \, v_n) / (\beta \Delta t^2) -
 * (1/(2\beta) - 1) a_n \f$ and the update above.
 *
 * The operators come from `specfem::linear_system`: \f$ K \f$ from the
 * `StiffnessAssembler` (Stacey-tolerant scope), \f$ C \f$ from the
 * `DampingAssembler` (empty on Stacey-free meshes), \f$ M \f$ from
 * `assemble_mass_vector`. \f$ A \f$ is constant for a fixed \f$ \Delta t \f$
 * and is assembled, fill-completed, and preconditioned once; each step is
 * one GMRES solve warm-started from \f$ u_n \f$ with an Ifpack2 right
 * preconditioner (true-residual convergence test). Without fluid-solid
 * coupling \f$ A \f$ is symmetric positive definite; GMRES is used so the
 * solver survives the non-symmetric coupling blocks planned next.
 * MueLu (AMG) is deferred: the float-only cluster Trilinos installs do not
 * link it (issue #1984).
 *
 * State \f$ (u, v, a) \f$ lives in Tpetra vectors; the assembly fields serve
 * as source-probe scratch and as the mirror that seismogram computation and
 * periodic tasks read. Run with a large dissipative step (see
 * @ref NewmarkBetaParameters::dissipative and
 * @ref ImplicitSolverConfig::steady_state_tolerance) the solver acts as a
 * static solver, recreating an explicit run driven to steady state in a few
 * solves.
 *
 * Scope (this milestone): serial, dim3, single-medium elastic isotropic,
 * NGLL = 5, no attenuation; boundaries `none`, `acoustic_free_surface`, or
 * `stacey`.
 *
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              dimension must be `dim3`; only `dim3, elastic, isotropic,
 *              none` is instantiated
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
class ImplicitNewmarkSolver : public solver {
public:
  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto medium_tag = Tags::medium_tag;

  using AssemblyType = specfem::assembly::assembly<dimension_tag>;

  /// Dof numbering and connectivity of the medium
  using MappingType =
      specfem::linear_system::FEMapping<dimension_tag, medium_tag>;

  /// Maps and sparsity graphs built over @ref MappingType
  using FEAssemblyType = specfem::linear_system::FEAssembly<MappingType>;
  using multivector_type =
      Tpetra::MultiVector<specfem::linear_system::scalar_type>;
  using operator_type = Tpetra::Operator<specfem::linear_system::scalar_type>;

  /**
   * @brief Assemble the operators and set up the linear solver.
   *
   * Assembles \f$ K \f$, \f$ C \f$, and \f$ M \f$, forms \f$ A \f$ for
   * `time_scheme->get_timestep()`, computes the Ifpack2 preconditioner, and
   * builds the Belos GMRES problem. Throws `std::runtime_error` outside the
   * supported scope (see the class docs) or for `beta <= 0` (the explicit
   * limit has no displacement-form operator).
   *
   * @param time_scheme Supplies the time step, step count, and seismogram
   *        cadence. Its predictor/corrector phases are NOT used -- the
   *        implicit update replaces them; \f$ \beta, \gamma \f$ come from
   *        `config`.
   * @param tasks Periodic tasks executed during the run (plotting, output)
   * @param assembly Spectral element assembly (shallow view-handle copy,
   *        like the explicit solver); fields are used as probe scratch and
   *        output mirror
   * @param config Newmark parameters, GMRES/preconditioner settings, and the
   *        optional steady-state stopping tolerance
   */
  ImplicitNewmarkSolver(
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<dimension_tag>>> &tasks,
      AssemblyType assembly, const ImplicitSolverConfig &config = {});

  /**
   * @brief Execute the implicit time loop.
   *
   * Per step: extract the source vector, form the right-hand side from the
   * previous state, solve \f$ A u_{n+1} = b \f$ (throws on GMRES
   * non-convergence), recover \f$ a_{n+1}, v_{n+1} \f$, write the state back
   * to the assembly fields, and run seismogram computation and periodic
   * tasks on the explicit solver's cadence. Stops early when the
   * steady-state criterion fires (if enabled).
   */
  void run() override;

  /// Dof maps and sparsity graphs shared by every assembled operator
  const FEAssemblyType &fe() const { return *fe_; }
  /// Assembled stiffness matrix \f$ K \f$
  Teuchos::RCP<const specfem::linear_system::crs_matrix_type>
  stiffness() const {
    return stiffness_;
  }
  /// Assembled Stacey damping matrix \f$ C \f$ (empty without Stacey)
  Teuchos::RCP<const specfem::linear_system::crs_matrix_type> damping() const {
    return damping_;
  }
  /// Lumped mass vector \f$ M \f$
  Teuchos::RCP<const specfem::linear_system::vector_type> mass() const {
    return mass_;
  }
  /// Implicit Newmark operator \f$ A \f$ for the configured time step
  Teuchos::RCP<const specfem::linear_system::crs_matrix_type>
  system_operator() const {
    return system_operator_;
  }
  /// Steps actually executed by the last run() (equals the step count unless
  /// the steady-state criterion stopped the run early)
  int last_step() const { return last_step_; }

  /**
   * @brief Change the steady-state stopping tolerance between runs.
   *
   * `run()` re-initializes the state, so one solver instance -- with its
   * assembled operators and preconditioner -- can be rerun under a
   * different stopping criterion without paying the assembly cost again.
   *
   * @param tolerance New tolerance (see
   *        @ref ImplicitSolverConfig::steady_state_tolerance); 0 disables
   */
  void set_steady_state_tolerance(const type_real tolerance) {
    config_.steady_state_tolerance = tolerance;
  }

private:
  /// (Re)build A from K, C, M for time step `dt`, then preconditioner and
  /// Belos problem. Idempotent; exact for any dt since K, C, M are kept.
  void form_operator(const type_real dt);

  /// Zero the acceleration field, run the production source kernel at
  /// `istep`, and gather the result: f = source vector at t_{n+1}
  void extract_source_vector(const int istep,
                             specfem::linear_system::vector_type &f);

  /// Scatter (u, v, a) into the assembly's forward field (host views, then
  /// device) so seismograms and periodic tasks see the current state
  void write_state_to_fields();

  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme_;
  std::vector<
      std::shared_ptr<specfem::periodic_tasks::periodic_task<dimension_tag>>>
      tasks_;                   ///< Periodic tasks
  AssemblyType assembly_;       ///< Assembly (probe scratch + output mirror)
  ImplicitSolverConfig config_; ///< Solver configuration

  /// Dof maps and sparsity graphs, built once and shared by the assemblers
  std::unique_ptr<FEAssemblyType> fe_;
  Teuchos::RCP<specfem::linear_system::crs_matrix_type> stiffness_;       ///< K
  Teuchos::RCP<specfem::linear_system::crs_matrix_type> damping_;         ///< C
  Teuchos::RCP<specfem::linear_system::vector_type> mass_;                ///< M
  Teuchos::RCP<specfem::linear_system::crs_matrix_type> system_operator_; ///< A

  Teuchos::RCP<operator_type> preconditioner_; ///< Ifpack2 right prec
  Teuchos::RCP<Belos::LinearProblem<specfem::linear_system::scalar_type,
                                    multivector_type, operator_type>>
      problem_; ///< Belos linear problem A u_new = b
  Teuchos::RCP<Belos::SolverManager<specfem::linear_system::scalar_type,
                                    multivector_type, operator_type>>
      gmres_; ///< Belos GMRES solver manager

  Teuchos::RCP<specfem::linear_system::vector_type> u_;     ///< u_n
  Teuchos::RCP<specfem::linear_system::vector_type> v_;     ///< v_n
  Teuchos::RCP<specfem::linear_system::vector_type> a_;     ///< a_n
  Teuchos::RCP<specfem::linear_system::vector_type> u_new_; ///< u_{n+1}
  Teuchos::RCP<specfem::linear_system::vector_type> a_new_; ///< a_{n+1}
  Teuchos::RCP<specfem::linear_system::vector_type> v_new_; ///< v_{n+1}
  Teuchos::RCP<specfem::linear_system::vector_type> rhs_;   ///< b
  Teuchos::RCP<specfem::linear_system::vector_type> tmp_;   ///< scratch
  Teuchos::RCP<specfem::linear_system::vector_type> tmp2_;  ///< scratch

  int last_step_ = 0; ///< Steps executed by the last run()
};

} // namespace solver
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
