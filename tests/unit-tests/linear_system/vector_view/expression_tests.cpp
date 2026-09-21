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

/**
 * @brief Every form the one-product grammar used to turn away.
 *
 * An expression is a term list plus any number of products, so a product
 * composes like any other operand. Each case is checked against its closed
 * form on a scaled-identity operator.
 */
TEST_F(VectorExpression, ProductsComposeLikeAnyOtherOperand) {
  auto u = space_->vector();
  auto v = space_->vector();
  auto f = space_->vector();
  auto m = space_->vector();
  auto b = space_->vector();
  const auto a_matrix = scaled_identity(static_cast<scalar_type>(2));
  const auto b_matrix = scaled_identity(static_cast<scalar_type>(3));
  const auto &A = *a_matrix;
  const auto &B = *b_matrix;

  const auto reset = [&]() {
    fill(u.vector(), [](int i) { return i + 1; });
    fill(v.vector(), [](int i) { return 10 * (i + 1); });
    fill(f.vector(), [](int i) { return 100 * (i + 1); });
    fill(m.vector(), [](int) { return 2; });
  };
  const auto expect = [&](const scalar_type per_dof) {
    std::vector<scalar_type> expected(num_dofs);
    for (int i = 0; i < num_dofs; ++i) {
      expected[i] = per_dof * static_cast<scalar_type>(i + 1);
    }
    expect_entries(b.vector(), expected);
  };

  // An expression carrying a product is itself an operand: scalable,
  // negatable, and addable -- none of which the one-product form allowed.
  reset();
  b = static_cast<scalar_type>(2) * (f - A * u);
  expect(196); // 2 * (100n - 2n)

  reset();
  b = -(f - A * u);
  expect(-98);

  reset();
  b = (f - A * u) + f;
  expect(198);

  // More than one product, of either kind, in one expression.
  reset();
  b = A * u + B * v;
  expect(32); // 2n + 30n

  reset();
  b = A * u + specfem::linear_system::diag(m) * v;
  expect(22); // 2n + 2 * 10n

  reset();
  b = f - A * u - B * v;
  expect(68); // 100n - 2n - 30n

  // Products over multi-term operands, and products scaled from outside.
  reset();
  b = A * (u + v) + B * (u - v);
  expect(-5); // 2 * 11n - 3 * 9n

  reset();
  b = static_cast<scalar_type>(2) * (A * u) +
      static_cast<scalar_type>(3) * (B * v);
  expect(94); // 4n + 90n

  // A nested product: the inner product is materialised, the outer applied.
  reset();
  b = A * (B * u);
  expect(6);
}

TEST_F(VectorExpression, AScaledOperatorFoldsIntoTheOperand) {
  using specfem::linear_system::operator*;

  auto u = space_->vector();
  auto m = space_->vector();
  auto b = space_->vector();
  const auto matrix = scaled_identity(static_cast<scalar_type>(2));

  fill(u.vector(), [](int i) { return i + 1; });
  fill(m.vector(), [](int) { return 2; });

  std::vector<scalar_type> expected(num_dofs);
  for (int i = 0; i < num_dofs; ++i) {
    expected[i] = static_cast<scalar_type>(6 * (i + 1));
  }

  // The coefficient reaches the alpha that apply() and elementWiseMultiply()
  // already take, so neither spelling costs a scaling pass.
  b = static_cast<scalar_type>(3) * (*matrix) * u;
  expect_entries(b.vector(), expected);

  b = static_cast<scalar_type>(3) * specfem::linear_system::diag(m) * u;
  expect_entries(b.vector(), expected);
}

TEST_F(VectorExpression, ANormOfAResidualWithASourceTerm) {
  auto u = space_->vector();
  auto f = space_->vector();
  const auto matrix = scaled_identity(static_cast<scalar_type>(2));

  // The expression that motivated generalising the grammar: an ordinary
  // residual with a source term, which the one-product form could not spell.
  fill(u.vector(), [](int i) { return i + 1; });
  fill(f.vector(), [](int i) { return i + 1; });

  // (f - A u + f) = (n - 2n + n) = 0
  EXPECT_NEAR(specfem::linear_system::norm2(f - *matrix * u + f), 0.0, 1e-4);
}

TEST_F(VectorExpression, ScratchIsBoundedByNestingNotByUse) {
  auto u = space_->vector();
  auto v = space_->vector();
  auto b = space_->vector();
  const auto a_matrix = scaled_identity(static_cast<scalar_type>(2));
  const auto b_matrix = scaled_identity(static_cast<scalar_type>(3));

  fill(u.vector(), [](int i) { return i + 1; });
  fill(v.vector(), [](int i) { return 10 * (i + 1); });

  // Two products side by side still need only one scratch vector at a time:
  // each is materialised, applied, and released before the next begins.
  for (int repeat = 0; repeat < 10; ++repeat) {
    b = *a_matrix * (u + v) + *b_matrix * (u - v);
  }
  EXPECT_EQ(space_->scratch_size(), std::size_t{ 1 });

  // A nested product is the one shape that borrows twice, because the inner
  // product must exist somewhere before the outer can be applied to it.
  for (int repeat = 0; repeat < 10; ++repeat) {
    const auto norm =
        specfem::linear_system::norm2(b - *a_matrix * (*b_matrix * u));
    (void)norm;
  }
  EXPECT_EQ(space_->scratch_size(), std::size_t{ 2 });

  // Either way the pool is sized by the deepest expression, never by how many
  // times it is evaluated -- nothing allocates per statement.
  const auto settled = space_->scratch_size();
  for (int repeat = 0; repeat < 50; ++repeat) {
    b = *a_matrix * (u + v) + *b_matrix * (u - v);
    const auto norm =
        specfem::linear_system::norm2(b - *a_matrix * (*b_matrix * u));
    (void)norm;
  }
  EXPECT_EQ(space_->scratch_size(), settled);
}

// Forms outside vector algebra remain ill-formed, enforced by the operand
// concept rather than by any rejection overload:
//
//   b = (*a_matrix) * (*b_matrix);  // matrix times matrix
//   b = diag(m) * (*a_matrix);      // diagonal times matrix
//   b = u + 1.0f;                   // vector plus scalar
//   b = u * v;                      // vector times vector
//   VectorView c = u;               // copy construction is deleted: `b = u`
//                                   // copies values, so a handle-sharing copy
//                                   // constructor would mean the opposite

} // namespace vector_view_expression_test

#else

TEST(VectorExpression, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM_ENABLE_TRILINOS is off; the vector expression "
                  "grammar is not built.";
}

#endif // SPECFEM_ENABLE_TRILINOS
