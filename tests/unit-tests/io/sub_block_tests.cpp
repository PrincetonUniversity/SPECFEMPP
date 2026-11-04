#include "SPECFEM_Environment.hpp"
#include "layout_oracle.hpp"
#include "specfem/datatype/domain_view.hpp"
#include "specfem/io.hpp"
#include "specfem/io/impl/sub_block.hpp"
#include "specfem/io_backends/impl/file_extents.hpp"
#include "specfem/parallel_configuration/chunk_config.hpp"
#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <utility>

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
            EXPECT_EQ(raw.at({ e, iz, ix }), value3(offset + e, iz, ix))
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
            EXPECT_EQ(raw.at({ e, iz, iy, ix }), value4(e, iz, iy, ix));
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
        EXPECT_EQ(dst(e, iz, ix), expected) << e << "," << iz << "," << ix;
      }
}

// Datasets must reject tiled DomainViews at compile time (including mirrors).
using DomainView2dHost =
    specfem::datatype::DomainView2d<double, 3, Kokkos::HostSpace>;
using DomainViewMirror = decltype(specfem::datatype::create_mirror_view(
    std::declval<DomainView2dHost>()));

static_assert(
    specfem::io_backends::impl::is_domain_view<FixedChunkView<3>>::value);
static_assert(
    specfem::io_backends::impl::is_domain_view<DomainView2dHost>::value);
static_assert(specfem::io_backends::impl::is_domain_view<
              DomainView2dHost::host_mirror_type>::value);
static_assert(
    specfem::io_backends::impl::is_domain_view<DomainViewMirror>::value);
static_assert(!specfem::io_backends::impl::is_domain_view<Kokkos::View<
                  double ***, Kokkos::LayoutLeft, Kokkos::HostSpace>>::value);
static_assert(!specfem::io_backends::impl::is_domain_view<
              DomainView2dHost::base_type>::value);

TEST(SubBlockTest, StorageChunkSizeIsReported) {
  const int chunk = specfem::parallel_configuration::storage_chunk_size;
  RecordProperty("storage_chunk_size", chunk);
  std::cout << "storage_chunk_size = " << chunk << std::endl;
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
