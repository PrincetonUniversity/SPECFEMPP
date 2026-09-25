#include "SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/tpetra_types.hpp"
#include "specfem/linear_system/vector_view/vector_view.hpp"
#include <Teuchos_RCP.hpp>
#include <Tpetra_Access.hpp>
#include <Tpetra_Core.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace vector_view_expression_test {

using scalar_type = specfem::linear_system::scalar_type;
using vector_type = specfem::linear_system::vector_type;
using map_type = specfem::linear_system::map_type;
using crs_matrix_type = specfem::linear_system::crs_matrix_type;
using VectorSpace = specfem::linear_system::VectorSpace;
using VectorView = specfem::linear_system::VectorView;

constexpr int num_dofs = 12;

/// Entries of a vector, gathered to the host
std::vector<scalar_type> entries(const vector_type &vector) {
  const auto local = vector.getLocalViewHost(Tpetra::Access::ReadOnly);
  std::vector<scalar_type> values(local.extent(0));
  for (std::size_t i = 0; i < local.extent(0); ++i) {
    values[i] = local(i, 0);
  }
  return values;
}

/// Overwrite a vector with `f(i)`
template <typename Function> void fill(vector_type &vector, Function f) {
  auto local = vector.getLocalViewHost(Tpetra::Access::OverwriteAll);
  for (std::size_t i = 0; i < local.extent(0); ++i) {
    local(i, 0) = static_cast<scalar_type>(f(static_cast<int>(i)));
  }
}

void expect_entries(const vector_type &vector,
                    const std::vector<scalar_type> &expected) {
  const auto actual = entries(vector);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto tolerance =
        static_cast<scalar_type>(1e-4) *
        (static_cast<scalar_type>(1) + std::abs(expected[i]));
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "entry " << i;
  }
}

/**
 * @brief A vector space over a contiguous map, with no mesh behind it.
 *
 * The grammar is mesh-independent, so these tests build their map directly
 * rather than assembling a simulation -- they run in milliseconds, unlike the
 * sparse_matrix_view fixtures.
 */
class VectorExpression : public ::testing::Test {
protected:
  void SetUp() override {
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() > 1) {
      GTEST_SKIP() << "the expression tests are single-rank.";
    }
    map_ = Teuchos::rcp(
        new map_type(static_cast<Tpetra::global_size_t>(num_dofs), 0, comm));
    space_ = std::make_unique<VectorSpace>(map_);
  }

  /// `alpha * I`, so that `A * x` has an obvious closed form
  Teuchos::RCP<crs_matrix_type> scaled_identity(const scalar_type alpha) {
    auto matrix = Teuchos::rcp(new crs_matrix_type(map_, 1));
    for (int row = 0; row < num_dofs; ++row) {
      const specfem::linear_system::global_ordinal_type global = row;
      matrix->insertGlobalValues(global, 1, &alpha, &global);
    }
    matrix->fillComplete();
    return matrix;
  }

  Teuchos::RCP<const map_type> map_;
  std::unique_ptr<VectorSpace> space_;
};

TEST_F(VectorExpression, ScalarAssignmentOverwritesEveryEntry) {
  auto u = space_->vector();
  u = static_cast<scalar_type>(3);
  expect_entries(u.vector(), std::vector<scalar_type>(num_dofs, 3));
}

TEST_F(VectorExpression, SumsOfEveryLengthMatchTheHandWrittenArithmetic) {
  auto u = space_->vector();
  auto v = space_->vector();
  auto a = space_->vector();
  auto b = space_->vector();

  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });
  fill(a.vector(), [](int i) { return 100 * (i + 1); });

  std::vector<scalar_type> expected(num_dofs);

  // One term
  b = static_cast<scalar_type>(2) * u;
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(2 * (i + 1));
  }
  expect_entries(b.vector(), expected);

  // Two terms -- one five-argument update
  b = static_cast<scalar_type>(2) * u + static_cast<scalar_type>(3) * v;
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(2 * (i + 1) + 30 * (i + 1));
  }
  expect_entries(b.vector(), expected);

  // Three terms -- the odd term takes the three-argument update
  b = static_cast<scalar_type>(2) * u + static_cast<scalar_type>(3) * v +
      static_cast<scalar_type>(4) * a;
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] =
        static_cast<scalar_type>(2 * (i + 1) + 30 * (i + 1) + 400 * (i + 1));
  }
  expect_entries(b.vector(), expected);

  // Four and five terms -- the pairwise emission wraps around
  b = u + v + a + u;
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(112 * (i + 1));
  }
  expect_entries(b.vector(), expected);

  b = u + v + a + u + v;
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(122 * (i + 1));
  }
  expect_entries(b.vector(), expected);
}

TEST_F(VectorExpression, SubtractionAndNegationFoldIntoCoefficients) {
  auto u = space_->vector();
  auto v = space_->vector();
  auto b = space_->vector();

  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });

  std::vector<scalar_type> expected(num_dofs);

  b = u - v;
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(-9 * (i + 1));
  }
  expect_entries(b.vector(), expected);

  b = -u;
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(-(i + 1));
  }
  expect_entries(b.vector(), expected);
}

TEST_F(VectorExpression, AccumulationNeverOverwrites) {
  auto u = space_->vector();
  auto v = space_->vector();
  auto b = space_->vector();

  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });

  b = static_cast<scalar_type>(0);
  b += u;
  b += v;
  b += static_cast<scalar_type>(2) * u;

  std::vector<scalar_type> expected(num_dofs);
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(13 * (i + 1));
  }
  expect_entries(b.vector(), expected);
}

TEST_F(VectorExpression, AssignmentFromAVectorCopiesValuesNotIdentity) {
  auto u = space_->vector();
  auto b = space_->vector();

  fill(u.vector(), [](int i) { return i + 1; });
  b = u;

  // Writing through b must not disturb u.
  b += u;
  std::vector<scalar_type> expected(num_dofs);
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(i + 1);
  }
  expect_entries(u.vector(), expected);
}

TEST_F(VectorExpression, MatrixProductAppliesTheOperator) {
  auto u = space_->vector();
  auto v = space_->vector();
  auto b = space_->vector();
  const auto matrix = scaled_identity(static_cast<scalar_type>(2));

  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });

  std::vector<scalar_type> expected(num_dofs);

  // A single unscaled operand is applied directly, without scratch
  b = *matrix * u;
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(2 * (i + 1));
  }
  expect_entries(b.vector(), expected);

  // A sum operand is materialised first, and the product accumulates
  b = static_cast<scalar_type>(0);
  b += *matrix *
       (static_cast<scalar_type>(2) * u + static_cast<scalar_type>(3) * v);
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(2 * (2 * (i + 1) + 30 * (i + 1)));
  }
  expect_entries(b.vector(), expected);
}

TEST_F(VectorExpression, DiagonalProductScalesEntrywise) {
  auto u = space_->vector();
  auto v = space_->vector();
  auto m = space_->vector();
  auto b = space_->vector();

  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });
  fill(m.vector(), [](int i) { return 2 * (i + 1); });

  b = static_cast<scalar_type>(0);
  b += specfem::linear_system::diag(m.vector()) * (u + v);

  std::vector<scalar_type> expected(num_dofs);
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(2 * (i + 1) * 11 * (i + 1));
  }
  expect_entries(b.vector(), expected);
}

TEST_F(VectorExpression, TheNewmarkRecoveryMatchesItsClosedForm) {
  auto u_new = space_->vector();
  auto u = space_->vector();
  auto v = space_->vector();
  auto a = space_->vector();
  auto a_new = space_->vector();

  fill(u_new.vector(), [](int i) { return 5 * (i + 1); });
  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });
  fill(a.vector(), [](int i) { return 100 * (i + 1); });

  const scalar_type c_a0 = 4;
  const scalar_type c_a2 = static_cast<scalar_type>(0.25);
  const scalar_type dt = static_cast<scalar_type>(0.5);

  a_new = c_a0 * (u_new - u - dt * v) - c_a2 * a;

  std::vector<scalar_type> expected(num_dofs);
  for (int i = 0; i < num_dofs; ++i) {
    const auto n = static_cast<scalar_type>(i + 1);
    expected[i] = c_a0 * (5 * n - n - dt * 10 * n) - c_a2 * 100 * n;
  }
  expect_entries(a_new.vector(), expected);
}

TEST_F(VectorExpression, AnAliasedTargetIsEvaluatedThroughScratch) {
  auto u = space_->vector();
  auto v = space_->vector();

  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });

  // A left-to-right emission would read a half-written u for the second term.
  u = static_cast<scalar_type>(2) * u + v;

  std::vector<scalar_type> expected(num_dofs);
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(12 * (i + 1));
  }
  expect_entries(u.vector(), expected);
}

TEST_F(VectorExpression, AccumulatingAnExpressionThatReadsTheTargetIsCorrect) {
  auto b = space_->vector();
  const auto matrix = scaled_identity(static_cast<scalar_type>(2));

  // Tpetra's apply() forbids its input aliasing its output, so this has to be
  // routed through scratch rather than handed straight to the operator.
  fill(b.vector(), [](int i) { return i + 1; });
  b += *matrix * b;

  std::vector<scalar_type> expected(num_dofs);
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(3 * (i + 1));
  }
  expect_entries(b.vector(), expected);

  fill(b.vector(), [](int i) { return i + 1; });
  b += static_cast<scalar_type>(2) * b;
  expect_entries(b.vector(), expected);
}

TEST_F(VectorExpression, Norm2OfAnExpressionLeavesItsOperandsAlone) {
  auto u = space_->vector();
  auto v = space_->vector();
  auto b = space_->vector();
  const auto matrix = scaled_identity(static_cast<scalar_type>(2));

  fill(u.vector(), [](int i) { return i == 0 ? 3 : (i == 1 ? 4 : 0); });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });

  const auto before = entries(u.vector());
  EXPECT_NEAR(specfem::linear_system::norm2(u), 5.0, 1e-4);
  EXPECT_NEAR(
      specfem::linear_system::norm2(u - static_cast<scalar_type>(0) * v), 5.0,
      1e-4);
  expect_entries(u.vector(), before);

  // A residual is one statement: b - A u, with A = 2I and b = 2u
  fill(b.vector(), [](int i) { return i == 0 ? 6 : (i == 1 ? 8 : 0); });
  EXPECT_NEAR(specfem::linear_system::norm2(b - *matrix * u), 0.0, 1e-4);
}

TEST_F(VectorExpression, SwapExchangesIdentitiesWithoutCopying) {
  auto u = space_->vector();
  auto v = space_->vector();

  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });

  const auto *u_storage = &u.vector();
  const auto *v_storage = &v.vector();

  swap(u, v);

  EXPECT_EQ(&u.vector(), v_storage);
  EXPECT_EQ(&v.vector(), u_storage);
}

TEST_F(VectorExpression, ScratchIsPooledAcrossStatements) {
  auto u = space_->vector();
  auto v = space_->vector();
  auto b = space_->vector();
  const auto matrix = scaled_identity(static_cast<scalar_type>(2));

  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });

  for (int repeat = 0; repeat < 10; ++repeat) {
    b = static_cast<scalar_type>(0);
    b += *matrix * (u + v);
    b += specfem::linear_system::diag(u.vector()) * (u + v);
    const auto norm = specfem::linear_system::norm2(b - *matrix * u);
    (void)norm;
  }

  // Borrows never interleave, so the pool holds exactly what the deepest
  // expression needs -- one vector -- however many times it is evaluated.
  EXPECT_LE(space_->scratch_size(), std::size_t{ 1 });
}

// The grammar rejects these at compile time; each is kept here so the intent
// is recorded next to the cases that must keep working.
//
//   b = 2.0f * (*matrix) * u;   // scaled matrix: scale the operand instead
//   b = *matrix * (*matrix * u); // nested product
//   b = *matrix * u + *matrix * v; // two products in one expression
//   b = *matrix * u - *matrix * v; // likewise, via the shared operator-
//   VectorView c = u;              // copy construction is deleted: `b = u`
//                                  // copies values, so a handle-sharing copy
//                                  // constructor would mean the opposite

} // namespace vector_view_expression_test

#else

TEST(VectorExpression, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM_ENABLE_TRILINOS is off; the vector expression "
                  "grammar is not built.";
}

#endif // SPECFEM_ENABLE_TRILINOS
