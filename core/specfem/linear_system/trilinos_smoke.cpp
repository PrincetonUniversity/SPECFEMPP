#include "specfem/linear_system/trilinos_smoke.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/setup.hpp"
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Ifpack2_Factory.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Tuple.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

int specfem::linear_system::trilinos_smoke_test() {
  using map_type = Tpetra::Map<>;
  // Float scalar: the Trilinos install instantiates single precision only
  // (Tpetra_INST_FLOAT=ON, Tpetra_INST_DOUBLE=OFF), matching type_real = float.
  using crs_matrix_type = Tpetra::CrsMatrix<float>;
  using global_ordinal_type = map_type::global_ordinal_type;
  using scalar_type = crs_matrix_type::scalar_type;

  const auto comm = Tpetra::getDefaultComm();

  const Tpetra::global_size_t num_global_entries = 1;
  const global_ordinal_type index_base = 0;
  const Teuchos::RCP<const map_type> row_map(
      new map_type(num_global_entries, index_base, comm));

  crs_matrix_type matrix(row_map, /*maxNumEntriesPerRow=*/1);
  const global_ordinal_type gid = 0;
  matrix.insertGlobalValues(gid, Teuchos::tuple<global_ordinal_type>(gid),
                            Teuchos::tuple<scalar_type>(scalar_type(1)));
  matrix.fillComplete();

  return static_cast<int>(matrix.getGlobalNumRows());
}

int specfem::linear_system::linear_solver_smoke_test() {
  using scalar_type = type_real;
  using map_type = Tpetra::Map<>;
  using crs_matrix_type = Tpetra::CrsMatrix<scalar_type>;
  using multivector_type = Tpetra::MultiVector<scalar_type>;
  using vector_type = Tpetra::Vector<scalar_type>;
  using operator_type = Tpetra::Operator<scalar_type>;
  using row_matrix_type =
      Tpetra::RowMatrix<scalar_type, crs_matrix_type::local_ordinal_type,
                        crs_matrix_type::global_ordinal_type,
                        crs_matrix_type::node_type>;
  using global_ordinal_type = map_type::global_ordinal_type;

  const auto comm = Tpetra::getDefaultComm();

  // SPD tridiagonal [-1, 2, -1] system, small enough to converge in a
  // handful of iterations yet exercising a genuine RILUK factorization.
  const Tpetra::global_size_t n = 16;
  const Teuchos::RCP<const map_type> map(
      new map_type(n, /*index_base=*/0, comm));

  auto matrix = Teuchos::rcp(new crs_matrix_type(map, /*maxEntriesPerRow=*/3));
  for (global_ordinal_type row = 0; row < static_cast<global_ordinal_type>(n);
       ++row) {
    if (row > 0) {
      matrix->insertGlobalValues(row, Teuchos::tuple(row - 1),
                                 Teuchos::tuple(scalar_type(-1)));
    }
    matrix->insertGlobalValues(row, Teuchos::tuple(row),
                               Teuchos::tuple(scalar_type(2)));
    if (row < static_cast<global_ordinal_type>(n) - 1) {
      matrix->insertGlobalValues(row, Teuchos::tuple(row + 1),
                                 Teuchos::tuple(scalar_type(-1)));
    }
  }
  matrix->fillComplete();

  // b = A * 1 so the exact solution is the vector of ones.
  auto x_exact = Teuchos::rcp(new vector_type(map));
  x_exact->putScalar(scalar_type(1));
  auto b = Teuchos::rcp(new vector_type(map));
  matrix->apply(*x_exact, *b);

  auto prec = Ifpack2::Factory::create<row_matrix_type>("RILUK", matrix);
  Teuchos::ParameterList prec_params;
  prec_params.set("fact: iluk level-of-fill", 1);
  prec->setParameters(prec_params);
  prec->initialize();
  prec->compute();

  auto x = Teuchos::rcp(new vector_type(map));
  auto problem = Teuchos::rcp(
      new Belos::LinearProblem<scalar_type, multivector_type, operator_type>(
          matrix, x, b));
  problem->setRightPrec(prec);
  problem->setProblem();

  auto belos_params = Teuchos::rcp(new Teuchos::ParameterList());
  belos_params->set("Convergence Tolerance", scalar_type(1e-5));
  belos_params->set("Maximum Iterations", 100);
  Belos::PseudoBlockGmresSolMgr<scalar_type, multivector_type, operator_type>
      solver(problem, belos_params);

  if (solver.solve() != Belos::Converged) {
    throw std::runtime_error(
        "specfem::linear_system::linear_solver_smoke_test: GMRES did not "
        "converge on the 16x16 tridiagonal system.");
  }

  x->update(scalar_type(-1), *x_exact, scalar_type(1));
  const auto error = x->norm2();
  if (!(error < scalar_type(1e-3))) {
    throw std::runtime_error(
        "specfem::linear_system::linear_solver_smoke_test: converged "
        "solution is wrong (||x - 1||_2 = " +
        std::to_string(error) + ").");
  }

  return solver.getNumIters();
}

#else

int specfem::linear_system::trilinos_smoke_test() { return 0; }

int specfem::linear_system::linear_solver_smoke_test() { return -1; }

#endif
