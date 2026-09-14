#include "SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/linear_system/sparse_matrix_view/fe_assembly.hpp"
#include "specfem/linear_system/sparse_matrix_view/matrix_view.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sparse_matrix_view_matrix_view_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using MappingType = specfem::linear_system::FEMapping<dim3_tag, elastic_tag>;
using FEAssemblyType = specfem::linear_system::FEAssembly<MappingType>;
using MatrixViewType = specfem::linear_system::SparseMatrixView<MappingType>;
using global_ordinal_type = specfem::linear_system::global_ordinal_type;
using scalar_type = specfem::linear_system::scalar_type;

/// Dense element blocks are LayoutRight, as the stiffness probe allocates them
using BlockRightType =
    Kokkos::View<scalar_type ***, Kokkos::LayoutRight, Kokkos::HostSpace>;
/// The same values in the opposite layout, to pin down layout independence
using BlockLeftType =
    Kokkos::View<scalar_type ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;

// Build a full assembly from a Newmark displacement-test dataset. Paths are
// relative to TEST_OUTPUT_DIR, where the displacement_tests data tree is
// linked (see SERIAL_LINK_DIRS in serial.cmake).
std::unique_ptr<AssemblyType> build_assembly_3d(const std::string &test_name) {
  const std::string test_path =
      "displacement_tests/Newmark/serial/dim3/" + test_name;

  specfem::runtime_configuration::setup setup(test_path +
                                              "/specfem_config.yaml");

  const auto database_filename = setup.get_databases();
  const auto &source_entries = setup.get_source_entries();
  const auto stations_node = setup.get_stations();
  const auto quadratures = setup.instantiate_quadrature();

  auto mesh = specfem::io::read_3d_mesh(database_filename,
                                        setup.get_attenuation_setup());

  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();

  auto [sources, t0, starttime] = specfem::io::read_sources<dim3_tag>(
      source_entries, nsteps, setup.get_t0(), dt, setup.get_simulation_type());
  (void)starttime;
  setup.update_t0(t0);

  auto receivers = specfem::io::read_3d_receivers(stations_node);

  return std::make_unique<AssemblyType>(
      mesh, quadratures, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, setup.get_max_seismogram_step(),
      setup.get_nstep_between_samples(), setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());
}

// A distinct, reproducible value per (element, row, column) so that a
// misordered or misplaced scatter cannot coincidentally agree.
scalar_type block_value(const int e, const int r, const int c) {
  return static_cast<scalar_type>(1 + e) * static_cast<scalar_type>(1 + r) +
         static_cast<scalar_type>(0.25) * static_cast<scalar_type>(1 + c);
}

// Every entry of a matrix, keyed by (row, column), for comparing two fills.
std::vector<std::pair<global_ordinal_type, scalar_type>>
row_entries(const specfem::linear_system::crs_matrix_type &matrix,
            const global_ordinal_type row) {
  using inds_type = specfem::linear_system::crs_matrix_type::
      nonconst_global_inds_host_view_type;
  using vals_type =
      specfem::linear_system::crs_matrix_type::nonconst_values_host_view_type;

  const std::size_t num_entries = matrix.getNumEntriesInGlobalRow(row);
  inds_type indices("indices", num_entries);
  vals_type values("values", num_entries);
  std::size_t num_returned = 0;
  matrix.getGlobalRowCopy(row, indices, values, num_returned);

  std::vector<std::pair<global_ordinal_type, scalar_type>> entries;
  entries.reserve(num_returned);
  for (std::size_t k = 0; k < num_returned; ++k) {
    entries.emplace_back(indices(k), values(k));
  }
  std::sort(entries.begin(), entries.end());
  return entries;
}

class SparseMatrixView3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    fe_.reset();
    delete assembly_;
    assembly_ = nullptr;
  }

  static AssemblyType &assembly() {
    if (assembly_ == nullptr) {
      assembly_ = build_assembly_3d("HomogeneousHalfspaceSmallNoABCForceSource")
                      .release();
    }
    return *assembly_;
  }

  static FEAssemblyType &fe() {
    if (!fe_) {
      fe_ = std::make_unique<FEAssemblyType>(MappingType(assembly()));
    }
    return *fe_;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<FEAssemblyType> fe_;
};

AssemblyType *SparseMatrixView3D::assembly_ = nullptr;
std::unique_ptr<FEAssemblyType> SparseMatrixView3D::fe_;

// ── DofSet: ordering ────────────────────────────────────────────────────────

// The contract the whole design rests on: a scalar-ispec dof set is indexed by
// local_dof_index, which is what orders the rows and columns of the element
// stiffness blocks. icomp is the SLOWEST of the trailing four indices, so the
// written slot order (ispec, iz, iy, ix, icomp) deliberately is not C-order.
TEST_F(SparseMatrixView3D, DofSetOrderMatchesLocalDofIndex) {
  const auto &mapping = fe().mapping();
  const int ispec = mapping.elements()(0);

  const auto dofs =
      mapping(ispec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  ASSERT_EQ(dofs.size(), mapping.ncomp() * mapping.ngllz() * mapping.nglly() *
                             mapping.ngllx());

  for (int icomp = 0; icomp < mapping.ncomp(); ++icomp) {
    for (int iz = 0; iz < mapping.ngllz(); ++iz) {
      for (int iy = 0; iy < mapping.nglly(); ++iy) {
        for (int ix = 0; ix < mapping.ngllx(); ++ix) {
          const int ldof =
              specfem::linear_system::local_dof_index<5>(icomp, iz, iy, ix);
          EXPECT_EQ(dofs[ldof], mapping(ispec, iz, iy, ix, icomp))
              << "dof set entry " << ldof << " disagrees with local_dof_index "
              << "at (icomp=" << icomp << ", iz=" << iz << ", iy=" << iy
              << ", ix=" << ix << ")";
        }
      }
    }
  }
}

// element_dofs is the pre-existing source of truth for the same ordering.
TEST_F(SparseMatrixView3D, DofSetExpandMatchesElementDofs) {
  const auto &mapping = fe().mapping();
  const int ispec = mapping.elements()(0);

  const auto dofs =
      mapping(ispec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  std::vector<global_ordinal_type> expanded(dofs.size());
  dofs.expand(expanded.begin());

  EXPECT_EQ(expanded, mapping.element_dofs(ispec));
}

// ── DofSet: selectors ───────────────────────────────────────────────────────

// A multi-element set is the concatenation of the per-element sets, not an
// interleaving -- which is what makes expand_block well defined.
TEST_F(SparseMatrixView3D, MultiElementDofSetConcatenatesPerElementSets) {
  const auto &mapping = fe().mapping();
  ASSERT_GE(mapping.nelements(), 3);

  const specfem::datatype::ElementIndexRange batch(mapping.elements()(0),
                                                   mapping.elements()(3));
  const auto dofs =
      mapping(batch, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  EXPECT_EQ(dofs.outer_extent(), 3);
  EXPECT_EQ(dofs.inner_size(), mapping.ncomp() * mapping.ngllz() *
                                   mapping.nglly() * mapping.ngllx());
  EXPECT_EQ(dofs.size(), 3 * dofs.inner_size());

  for (int e = 0; e < 3; ++e) {
    std::vector<global_ordinal_type> block(dofs.inner_size());
    dofs.expand_block(e, block.begin());
    EXPECT_EQ(block, mapping.element_dofs(batch(e)))
        << "block " << e << " of a multi-element set is not element "
        << batch(e) << "'s dof set";
  }
}

// A scalar slot narrows the set; ALL in the point slot spans the medium.
TEST_F(SparseMatrixView3D, MixedSelectorsHaveExpectedShape) {
  const auto &mapping = fe().mapping();

  // One mesh point, all components -- the shape of a damping block's rows.
  const auto point_dofs = mapping(7, Kokkos::ALL);
  ASSERT_EQ(point_dofs.size(), mapping.ncomp());
  for (int icomp = 0; icomp < mapping.ncomp(); ++icomp) {
    EXPECT_EQ(point_dofs[icomp], mapping(7, icomp));
  }

  // All points, one component.
  const auto comp_dofs = mapping(Kokkos::ALL, 1);
  ASSERT_EQ(comp_dofs.size(), mapping.nglob());
  EXPECT_EQ(comp_dofs[0], mapping(0, 1));
  EXPECT_EQ(comp_dofs[mapping.nglob() - 1], mapping(mapping.nglob() - 1, 1));

  // One element, one component, all GLL points.
  const int ispec = mapping.elements()(0);
  const auto slice = mapping(ispec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 2);
  ASSERT_EQ(slice.size(), mapping.ngllz() * mapping.nglly() * mapping.ngllx());
  EXPECT_EQ(slice[0], mapping(ispec, 0, 0, 0, 2));
}

// ── SparseMatrixView: rank dispatch and the extent rule ─────────────────────

// The assertion that the batched form in fill_matrix is safe: one rank-3
// update over N elements must equal N rank-2 updates with a scalar ispec.
TEST_F(SparseMatrixView3D, Rank3UpdateEqualsPerElementRank2Updates) {
  const auto &mapping = fe().mapping();
  ASSERT_GE(mapping.nelements(), 4);

  const int ndof_e =
      mapping.ncomp() * mapping.ngllz() * mapping.nglly() * mapping.ngllx();
  const int nblocks = 4;
  const specfem::datatype::ElementIndexRange batch(mapping.elements()(0),
                                                   mapping.elements()(nblocks));

  BlockRightType blocks("blocks", nblocks, ndof_e, ndof_e);
  for (int e = 0; e < nblocks; ++e) {
    for (int r = 0; r < ndof_e; ++r) {
      for (int c = 0; c < ndof_e; ++c) {
        blocks(e, r, c) = block_value(e, r, c);
      }
    }
  }

  // Batched: one rank-3 update.
  MatrixViewType batched(fe().full_matrix_graph(), mapping);
  batched.begin_fill();
  const auto batch_dofs =
      mapping(batch, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  batched(batch_dofs, batch_dofs) += blocks;
  batched.finalize();

  // Element by element: nblocks rank-2 updates.
  MatrixViewType per_element(fe().full_matrix_graph(), mapping);
  per_element.begin_fill();
  for (int e = 0; e < nblocks; ++e) {
    const auto dofs =
        mapping(batch(e), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    const auto block = Kokkos::subview(blocks, e, Kokkos::ALL, Kokkos::ALL);
    per_element(dofs, dofs) += block;
  }
  per_element.finalize();

  const auto n = mapping.num_global_dofs();
  for (global_ordinal_type row = 0; row < n; ++row) {
    EXPECT_EQ(row_entries(*batched.matrix(), row),
              row_entries(*per_element.matrix(), row))
        << "row " << row << " differs between the batched and per-element fill";
  }
}

// The partial-final-batch case: the probe's block buffer is allocated once at
// the full batch size and reused, so the last update carries more blocks than
// the dof set names. The dof set is the authority; the surplus is ignored.
TEST_F(SparseMatrixView3D, OverlongRank3BlockScattersOnlyTheNamedBlocks) {
  const auto &mapping = fe().mapping();
  ASSERT_GE(mapping.nelements(), 3);

  const int ndof_e =
      mapping.ncomp() * mapping.ngllz() * mapping.nglly() * mapping.ngllx();
  const int named = 2;
  const int allocated = 5;
  const specfem::datatype::ElementIndexRange batch(mapping.elements()(0),
                                                   mapping.elements()(named));

  BlockRightType oversized("oversized", allocated, ndof_e, ndof_e);
  BlockRightType exact("exact", named, ndof_e, ndof_e);
  for (int e = 0; e < allocated; ++e) {
    for (int r = 0; r < ndof_e; ++r) {
      for (int c = 0; c < ndof_e; ++c) {
        // Trailing blocks hold values that would be visible if scattered.
        oversized(e, r, c) = block_value(e, r, c);
        if (e < named) {
          exact(e, r, c) = block_value(e, r, c);
        }
      }
    }
  }

  const auto dofs =
      mapping(batch, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  MatrixViewType from_oversized(fe().full_matrix_graph(), mapping);
  from_oversized.begin_fill();
  from_oversized(dofs, dofs) += oversized;
  from_oversized.finalize();

  MatrixViewType from_exact(fe().full_matrix_graph(), mapping);
  from_exact.begin_fill();
  from_exact(dofs, dofs) += exact;
  from_exact.finalize();

  const auto n = mapping.num_global_dofs();
  for (global_ordinal_type row = 0; row < n; ++row) {
    EXPECT_EQ(row_entries(*from_oversized.matrix(), row),
              row_entries(*from_exact.matrix(), row))
        << "row " << row << ": the surplus blocks of an over-long update "
        << "were not ignored";
  }
}

// Too few blocks is a genuine shape error, unlike a surplus.
TEST_F(SparseMatrixView3D, TooFewBlocksThrows) {
  const auto &mapping = fe().mapping();
  ASSERT_GE(mapping.nelements(), 3);

  const int ndof_e =
      mapping.ncomp() * mapping.ngllz() * mapping.nglly() * mapping.ngllx();
  const specfem::datatype::ElementIndexRange batch(mapping.elements()(0),
                                                   mapping.elements()(3));
  const auto dofs =
      mapping(batch, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  BlockRightType too_few("too_few", 2, ndof_e, ndof_e);

  MatrixViewType matrix(fe().full_matrix_graph(), mapping);
  matrix.begin_fill();
  EXPECT_THROW(matrix(dofs, dofs) += too_few, std::runtime_error);
}

// An oversized inner extent is not a scratch buffer -- it is a block from a
// different NGLL or component count, and must not be silently truncated.
TEST_F(SparseMatrixView3D, MismatchedInnerExtentThrows) {
  const auto &mapping = fe().mapping();
  const int ndof_e =
      mapping.ncomp() * mapping.ngllz() * mapping.nglly() * mapping.ngllx();
  const specfem::datatype::ElementIndexRange batch(mapping.elements()(0),
                                                   mapping.elements()(1));
  const auto dofs =
      mapping(batch, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  BlockRightType too_wide("too_wide", 1, ndof_e, ndof_e + 1);

  MatrixViewType matrix(fe().full_matrix_graph(), mapping);
  matrix.begin_fill();
  EXPECT_THROW(matrix(dofs, dofs) += too_wide, std::runtime_error);
}

// ── SparseMatrixView: layout, accumulation, fill state ──────────────────────

// Layout is a property of the block's type, not an assumption about a pointer.
TEST_F(SparseMatrixView3D, LayoutLeftAndRightBlocksAgree) {
  const auto &mapping = fe().mapping();
  const int ndof_e =
      mapping.ncomp() * mapping.ngllz() * mapping.nglly() * mapping.ngllx();
  const specfem::datatype::ElementIndexRange batch(mapping.elements()(0),
                                                   mapping.elements()(2));
  const auto dofs =
      mapping(batch, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  BlockRightType right("right", 2, ndof_e, ndof_e);
  BlockLeftType left("left", 2, ndof_e, ndof_e);
  for (int e = 0; e < 2; ++e) {
    for (int r = 0; r < ndof_e; ++r) {
      for (int c = 0; c < ndof_e; ++c) {
        right(e, r, c) = block_value(e, r, c);
        left(e, r, c) = block_value(e, r, c);
      }
    }
  }

  MatrixViewType from_right(fe().full_matrix_graph(), mapping);
  from_right.begin_fill();
  from_right(dofs, dofs) += right;
  from_right.finalize();

  MatrixViewType from_left(fe().full_matrix_graph(), mapping);
  from_left.begin_fill();
  from_left(dofs, dofs) += left;
  from_left.finalize();

  const auto n = mapping.num_global_dofs();
  for (global_ordinal_type row = 0; row < n; ++row) {
    EXPECT_EQ(row_entries(*from_right.matrix(), row),
              row_entries(*from_left.matrix(), row))
        << "row " << row << " differs between LayoutRight and LayoutLeft";
  }
}

// Elements share mesh points, so assembly is accumulation rather than
// assignment: repeated updates to a region must sum into it.
TEST_F(SparseMatrixView3D, RepeatedUpdatesAccumulate) {
  const auto &mapping = fe().mapping();
  const int ndof_e =
      mapping.ncomp() * mapping.ngllz() * mapping.nglly() * mapping.ngllx();
  const int ispec = mapping.elements()(0);
  const auto dofs =
      mapping(ispec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  Kokkos::View<scalar_type **, Kokkos::LayoutRight, Kokkos::HostSpace> ones(
      "ones", ndof_e, ndof_e);
  Kokkos::deep_copy(ones, static_cast<scalar_type>(1));

  MatrixViewType matrix(fe().full_matrix_graph(), mapping);
  matrix.begin_fill();
  matrix(dofs, dofs) += ones;
  matrix(dofs, dofs) += ones;
  matrix.finalize();

  const auto entries = row_entries(*matrix.matrix(), dofs[0]);
  int nonzero = 0;
  for (const auto &[column, value] : entries) {
    (void)column;
    if (value != 0) {
      EXPECT_NEAR(value, static_cast<scalar_type>(2), 1e-5);
      ++nonzero;
    }
  }
  EXPECT_EQ(nonzero, ndof_e)
      << "the element's own dofs should each have accumulated twice";
}

// Tpetra's fill-active/fill-complete distinction stays visible: it has no
// MATLAB analogue and hiding it would turn a clear error into a silent one.
TEST_F(SparseMatrixView3D, UpdateBeforeBeginFillThrows) {
  const auto &mapping = fe().mapping();
  const int ispec = mapping.elements()(0);
  const auto dofs =
      mapping(ispec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  MatrixViewType matrix(fe().full_matrix_graph(), mapping);
  EXPECT_FALSE(matrix.is_fill_active());
  EXPECT_THROW(matrix(dofs, dofs), std::runtime_error);
}

// A block whose columns are outside the sparsity pattern must not be dropped
// silently, which is what the raw Tpetra call does.
TEST_F(SparseMatrixView3D, ColumnsOutsideTheGraphThrow) {
  const auto &mapping = fe().mapping();

  // The damping graph is block-diagonal, and empty on this no-ABC mesh, so an
  // element-dense block cannot fit inside it.
  const int ispec = mapping.elements()(0);
  const auto dofs =
      mapping(ispec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  const int ndof_e = dofs.size();

  Kokkos::View<scalar_type **, Kokkos::LayoutRight, Kokkos::HostSpace> ones(
      "ones", ndof_e, ndof_e);
  Kokkos::deep_copy(ones, static_cast<scalar_type>(1));

  MatrixViewType matrix(fe().damping_matrix_graph(), mapping);
  matrix.begin_fill();
  EXPECT_THROW(matrix(dofs, dofs) += ones, std::runtime_error);
}

// ── Matrix and diagonal sums ────────────────────────────────────────────────

// Fill a fresh view on the full graph with one element block, and return it.
Teuchos::RCP<specfem::linear_system::fe_crs_matrix_type>
filled_matrix(FEAssemblyType &fe, const scalar_type scale) {
  const auto &mapping = fe.mapping();
  const int ndof_e =
      mapping.ncomp() * mapping.ngllz() * mapping.nglly() * mapping.ngllx();
  const int ispec = mapping.elements()(0);

  Kokkos::View<scalar_type **, Kokkos::LayoutRight, Kokkos::HostSpace> block(
      "block", ndof_e, ndof_e);
  for (int r = 0; r < ndof_e; ++r) {
    for (int c = 0; c < ndof_e; ++c) {
      block(r, c) = scale * block_value(0, r, c);
    }
  }

  MatrixViewType view(fe.full_matrix_graph(), mapping);
  view.begin_fill();
  const auto dofs =
      mapping(ispec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  view(dofs, dofs) += block;
  view.finalize();
  return view.matrix();
}

// A += B on the same graph reproduces B exactly.
TEST_F(SparseMatrixView3D, AddingAMatrixReproducesIt) {
  const auto &mapping = fe().mapping();
  const auto source = filled_matrix(fe(), 1);

  MatrixViewType sum(fe().full_matrix_graph(), mapping);
  sum.begin_fill();
  sum += *source;
  sum.finalize();

  const auto n = mapping.num_global_dofs();
  for (global_ordinal_type row = 0; row < n; ++row) {
    ASSERT_EQ(row_entries(*sum.matrix(), row), row_entries(*source, row))
        << "row " << row;
  }
}

// A += alpha * B scales every entry, and agrees with a matrix built at that
// scale in the first place.
TEST_F(SparseMatrixView3D, AddingAScaledMatrixScalesEveryEntry) {
  using specfem::linear_system::operator*;
  const auto &mapping = fe().mapping();
  const scalar_type alpha = 3;

  const auto source = filled_matrix(fe(), 1);
  const auto prescaled = filled_matrix(fe(), alpha);

  MatrixViewType sum(fe().full_matrix_graph(), mapping);
  sum.begin_fill();
  sum += alpha * (*source);
  sum.finalize();

  const auto n = mapping.num_global_dofs();
  for (global_ordinal_type row = 0; row < n; ++row) {
    const auto scaled = row_entries(*sum.matrix(), row);
    const auto expected = row_entries(*prescaled, row);
    ASSERT_EQ(scaled.size(), expected.size()) << "row " << row;
    for (std::size_t k = 0; k < scaled.size(); ++k) {
      EXPECT_EQ(scaled[k].first, expected[k].first);
      EXPECT_NEAR(scaled[k].second, expected[k].second,
                  1e-3 * std::abs(expected[k].second) + 1e-5)
          << "row " << row << ", entry " << k;
    }
  }
}

// A += alpha * diag(v) touches the diagonal and nothing else.
TEST_F(SparseMatrixView3D, AddingAScaledDiagonalTouchesOnlyTheDiagonal) {
  using specfem::linear_system::diag;
  using specfem::linear_system::operator*;
  const auto &mapping = fe().mapping();
  const scalar_type alpha = 2;

  specfem::linear_system::vector_type v(fe().owned_map());
  {
    auto view = v.getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (std::size_t dof = 0; dof < view.extent(0); ++dof) {
      view(dof, 0) = static_cast<scalar_type>(1 + dof);
    }
  }

  MatrixViewType sum(fe().full_matrix_graph(), mapping);
  sum.begin_fill();
  sum += alpha * diag(v);
  sum.finalize();

  const auto n = mapping.num_global_dofs();
  for (global_ordinal_type row = 0; row < n; ++row) {
    for (const auto &[column, value] : row_entries(*sum.matrix(), row)) {
      if (column == row) {
        EXPECT_NEAR(value, alpha * static_cast<scalar_type>(1 + row),
                    1e-3 * std::abs(value) + 1e-5)
            << "diagonal entry of row " << row;
      } else {
        EXPECT_EQ(value, 0)
            << "off-diagonal (" << row << ", " << column << ") was touched";
      }
    }
  }
}

// The three terms of K + c1 C + c2 M are an accumulation, so their order must
// not matter.
TEST_F(SparseMatrixView3D, TermsAccumulateInAnyOrder) {
  using specfem::linear_system::diag;
  using specfem::linear_system::operator*;
  const auto &mapping = fe().mapping();

  const auto k = filled_matrix(fe(), 1);
  const auto c = filled_matrix(fe(), 2);
  specfem::linear_system::vector_type m(fe().owned_map());
  m.putScalar(static_cast<scalar_type>(0.5));

  MatrixViewType forward(fe().full_matrix_graph(), mapping);
  forward.begin_fill();
  forward += *k;
  forward += static_cast<scalar_type>(4) * (*c);
  forward += static_cast<scalar_type>(7) * diag(m);
  forward.finalize();

  MatrixViewType reversed(fe().full_matrix_graph(), mapping);
  reversed.begin_fill();
  reversed += static_cast<scalar_type>(7) * diag(m);
  reversed += static_cast<scalar_type>(4) * (*c);
  reversed += *k;
  reversed.finalize();

  const auto n = mapping.num_global_dofs();
  for (global_ordinal_type row = 0; row < n; ++row) {
    const auto a = row_entries(*forward.matrix(), row);
    const auto b = row_entries(*reversed.matrix(), row);
    ASSERT_EQ(a.size(), b.size()) << "row " << row;
    for (std::size_t j = 0; j < a.size(); ++j) {
      EXPECT_EQ(a[j].first, b[j].first);
      EXPECT_NEAR(a[j].second, b[j].second, 1e-3 * std::abs(a[j].second) + 1e-5)
          << "row " << row << ", entry " << j;
    }
  }
}

// Adding a matrix whose sparsity is not contained in the target's must not be
// silently truncated -- that is what the raw Tpetra call would do.
TEST_F(SparseMatrixView3D, AddingAMatrixOutsideTheGraphThrows) {
  const auto &mapping = fe().mapping();
  const auto source = filled_matrix(fe(), 1);

  // The damping graph is block-diagonal and empty on this no-ABC mesh, so an
  // element-dense matrix cannot fit inside it.
  MatrixViewType narrow(fe().damping_matrix_graph(), mapping);
  narrow.begin_fill();
  EXPECT_THROW(narrow += *source, std::runtime_error);
}

} // namespace sparse_matrix_view_matrix_view_test

#else

TEST(SparseMatrixView3D, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM_ENABLE_TRILINOS is off; the sparse matrix view is "
                  "not built.";
}

#endif // SPECFEM_ENABLE_TRILINOS
