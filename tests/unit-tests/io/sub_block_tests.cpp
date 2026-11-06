#include "SPECFEM_Environment.hpp"
#include "layout_oracle.hpp"
#include "specfem/datatype/domain_view.hpp"
#include "specfem/io.hpp"
#include "specfem/io/impl/sub_block.hpp"
#include "specfem/io_backends/impl/file_extents.hpp"
#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = boost::filesystem;

template <std::size_t Rank>
using FixedChunkView =
    specfem::datatype::View<double, Kokkos::dextents<std::size_t, Rank>,
                            specfem::datatype::DomainViewMapping<
                                specfem::element::dimension_tag::dim2, 4>,
                            Kokkos::HostSpace>;

// Removes a per-test output directory on scope exit.
class DirGuard {
public:
  explicit DirGuard(std::string path) : path_(std::move(path)) { remove(); }
  ~DirGuard() { remove(); }
  const std::string &path() const { return path_; }

private:
  void remove() {
    if (fs::exists(path_))
      fs::remove_all(path_);
  }
  std::string path_;
};

static double value3(std::size_t e, std::size_t iz, std::size_t ix) {
  return 1000.0 * e + 10.0 * iz + ix + 1.0;
}

static double value4(std::size_t e, std::size_t iz, std::size_t iy,
                     std::size_t ix) {
  return 1000.0 * e + 100.0 * iz + 10.0 * iy + ix + 1.0;
}

static void fill3(const FixedChunkView<3> &view) {
  for (std::size_t e = 0; e < view.extent(0); ++e)
    for (std::size_t iz = 0; iz < view.extent(1); ++iz)
      for (std::size_t ix = 0; ix < view.extent(2); ++ix)
        view(e, iz, ix) = value3(e, iz, ix);
}

TEST(SubBlockTest, ExtractSubBlockMatchesHeader) {
  for (const std::size_t nelem : { 5, 7, 8 }) {
    DirGuard guard("test_sub_block_extract_" + std::to_string(nelem));
    FixedChunkView<3> view("view", nelem, 3, 2);
    fill3(view);

    for (const auto [offset, count] :
         { std::pair<int, int>{ 0, static_cast<int>(nelem) },
           std::pair<int, int>{ 2, 3 } }) {
      if (offset + count > static_cast<int>(nelem))
        continue;
      {
        specfem::io_backends::NPY<specfem::io::write>::File file(guard.path());
        auto sub =
            specfem::io::impl::extract_sub_block(view, "k", offset, count);
        file.createDataset("k", sub).write();
        file.flush();
      }
      const auto raw = specfem_test::layout_oracle::read_npy_raw<double>(
          guard.path() + "/k.npy");
      EXPECT_FALSE(raw.fortran_order);
      const std::vector<std::size_t> expected = {
        static_cast<std::size_t>(count), 3, 2
      };
      ASSERT_EQ(raw.shape, expected);
      for (std::size_t e = 0; e < expected[0]; ++e)
        for (std::size_t iz = 0; iz < 3; ++iz)
          for (std::size_t ix = 0; ix < 2; ++ix)
            EXPECT_NEAR(raw.at({ e, iz, ix }), value3(offset + e, iz, ix),
                        1e-12)
                << "nelem=" << nelem << " offset=" << offset;
    }
  }
}

TEST(SubBlockTest, ExtractSubBlockRank4) {
  for (const std::size_t nelem : { 5, 7, 8 }) {
    DirGuard guard("test_sub_block_rank4_" + std::to_string(nelem));
    FixedChunkView<4> view("view", nelem, 3, 2, 2);
    for (std::size_t e = 0; e < nelem; ++e)
      for (std::size_t iz = 0; iz < 3; ++iz)
        for (std::size_t iy = 0; iy < 2; ++iy)
          for (std::size_t ix = 0; ix < 2; ++ix)
            view(e, iz, iy, ix) = value4(e, iz, iy, ix);

    {
      specfem::io_backends::NPY<specfem::io::write>::File file(guard.path());
      auto sub = specfem::io::impl::extract_sub_block(view, "k", 0,
                                                      static_cast<int>(nelem));
      file.createDataset("k", sub).write();
      file.flush();
    }
    const auto raw = specfem_test::layout_oracle::read_npy_raw<double>(
        guard.path() + "/k.npy");
    EXPECT_FALSE(raw.fortran_order);
    const std::vector<std::size_t> expected = { nelem, 3, 2, 2 };
    ASSERT_EQ(raw.shape, expected);
    for (std::size_t e = 0; e < nelem; ++e)
      for (std::size_t iz = 0; iz < 3; ++iz)
        for (std::size_t iy = 0; iy < 2; ++iy)
          for (std::size_t ix = 0; ix < 2; ++ix)
            EXPECT_NEAR(raw.at({ e, iz, iy, ix }), value4(e, iz, iy, ix),
                        1e-12);
  }
}

TEST(SubBlockTest, InsertSubBlockRoundTrip) {
  FixedChunkView<3> dst("dst", 7, 3, 2);
  for (std::size_t e = 0; e < 7; ++e)
    for (std::size_t iz = 0; iz < 3; ++iz)
      for (std::size_t ix = 0; ix < 2; ++ix)
        dst(e, iz, ix) = 0.0;

  const int offset = 1, count = 4;
  auto sub = specfem::io::impl::make_sub_block(dst, "k", count);
  for (int e = 0; e < count; ++e)
    for (std::size_t iz = 0; iz < 3; ++iz)
      for (std::size_t ix = 0; ix < 2; ++ix)
        sub(e, iz, ix) = value3(e, iz, ix);

  specfem::io::impl::insert_sub_block(dst, sub, offset);

  for (int e = 0; e < 7; ++e)
    for (std::size_t iz = 0; iz < 3; ++iz)
      for (std::size_t ix = 0; ix < 2; ++ix) {
        const bool inserted = e >= offset && e < offset + count;
        const double expected = inserted ? value3(e - offset, iz, ix) : 0.0;
        EXPECT_NEAR(dst(e, iz, ix), expected, 1e-12)
            << e << "," << iz << "," << ix;
      }
}

TEST(StorageOrderTest, DomainViewIsWritableOnlyAsOneFullTile) {
  EXPECT_FALSE(specfem::io_backends::impl::contiguous_storage_order(
      FixedChunkView<3>("padded", 7, 3, 2)));
  EXPECT_FALSE(specfem::io_backends::impl::contiguous_storage_order(
      FixedChunkView<3>("two_tiles", 8, 3, 2)));

  FixedChunkView<3> one_tile("one_tile", 4, 3, 2);
  EXPECT_EQ(specfem::io_backends::impl::contiguous_storage_order(one_tile),
            specfem::io_backends::impl::StorageOrder::column_major);
  for (std::size_t e = 0; e < 4; ++e)
    for (std::size_t iz = 0; iz < 3; ++iz)
      for (std::size_t ix = 0; ix < 2; ++ix)
        EXPECT_EQ(&one_tile(e, iz, ix), one_tile.data() + e + 4 * iz + 12 * ix);
}

TEST(StorageOrderTest, KokkosViewsAreClassifiedByStrides) {
  EXPECT_EQ(specfem::io_backends::impl::contiguous_storage_order(
                Kokkos::View<double ***, Kokkos::LayoutLeft, Kokkos::HostSpace>(
                    "left", 4, 3, 2)),
            specfem::io_backends::impl::StorageOrder::column_major);
  EXPECT_EQ(
      specfem::io_backends::impl::contiguous_storage_order(
          Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::HostSpace>(
              "right", 4, 3, 2)),
      specfem::io_backends::impl::StorageOrder::row_major);
  EXPECT_EQ(specfem::io_backends::impl::contiguous_storage_order(
                Kokkos::View<double *, Kokkos::HostSpace>("rank1", 5)),
            specfem::io_backends::impl::StorageOrder::row_major);
  // Both stride sets match a shape of all ones; row-major is reported.
  EXPECT_EQ(specfem::io_backends::impl::contiguous_storage_order(
                Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>(
                    "ones_left", 1, 1)),
            specfem::io_backends::impl::StorageOrder::row_major);

  using StridedView =
      Kokkos::View<double **, Kokkos::LayoutStride, Kokkos::HostSpace>;
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> matrix(
      "matrix", 6, 3);
  const StridedView every_other_row(matrix.data(),
                                    Kokkos::LayoutStride(3, 6, 3, 1));
  EXPECT_FALSE(
      specfem::io_backends::impl::contiguous_storage_order(every_other_row));
  const StridedView full_range(matrix.data(), Kokkos::LayoutStride(6, 3, 3, 1));
  EXPECT_EQ(specfem::io_backends::impl::contiguous_storage_order(full_range),
            specfem::io_backends::impl::StorageOrder::row_major);
}

TEST(StorageOrderTest, NPYDatasetRejectsPaddedDomainView) {
  DirGuard guard("test_storage_order_padded_domain_view");
  specfem::io_backends::NPY<specfem::io::write>::File file(guard.path());
  FixedChunkView<3> padded("padded", 7, 3, 2);
  EXPECT_THROW(file.createDataset("k", padded), std::invalid_argument);
}

// Writes a column-major NPY file by hand, as an external tool would.
static void write_fortran_order_npy(const std::string &path,
                                    const std::vector<std::size_t> &shape,
                                    const std::vector<double> &values) {
  const auto header =
      specfem::io_backends::impl::NPY::create_npy_header<double>(shape, true);
  std::ofstream stream(path, std::ios::binary);
  stream.write(header.data(), header.size());
  stream.write(reinterpret_cast<const char *>(values.data()),
               values.size() * sizeof(double));
}

TEST(StorageOrderTest, NPYReaderIgnoresOrderWhenOneExtentIsAboveOne) {
  DirGuard guard("test_storage_order_reader_ambiguous");
  fs::create_directories(guard.path());
  write_fortran_order_npy(guard.path() + "/ones.npy", { 1, 1 }, { 3.5 });
  write_fortran_order_npy(guard.path() + "/square.npy", { 2, 2 },
                          { 1.0, 2.0, 3.0, 4.0 });

  specfem::io_backends::NPY<specfem::io::read>::File file(guard.path());
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> ones("ones",
                                                                       1, 1);
  EXPECT_NO_THROW(file.openDataset("ones", ones).read());
  EXPECT_NEAR(ones(0, 0), 3.5, 1e-12);

  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> square(
      "square", 2, 2);
  EXPECT_THROW(file.openDataset("square", square).read(), std::runtime_error);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
