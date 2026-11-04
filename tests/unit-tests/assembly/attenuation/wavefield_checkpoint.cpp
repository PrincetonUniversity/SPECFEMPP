#include "../../io/layout_oracle.hpp"
#include "../test_fixture/test_fixture.hpp"
#include "specfem/io.hpp"
#include "specfem/io/wavefield/reader.hpp"
#include "specfem/io/wavefield/writer.hpp"
#include "specfem/parallel_configuration/chunk_config.hpp"
#include "specfem/tag_dispatch.hpp"
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

namespace wavefield_checkpoint_impl {

using WriteBackend = specfem::io_backends::NPY<specfem::io::write>;
using ReadBackend = specfem::io_backends::NPY<specfem::io::read>;
using AttenuationType =
    specfem::assembly::Attenuation<specfem::element::dimension_tag::dim2>;

constexpr int checkpoint_step = 1;

std::string temp_directory(const std::string &suffix) {
  return (std::getenv("BUILD_DIR")
              ? std::string(std::getenv("BUILD_DIR"))
              : boost::filesystem::current_path().string()) +
         "/tests/unit-tests/temp_wavefield_attenuation_" + suffix;
}

// Calls f(dataset_name, host_view) for every checkpointed attenuation view.
template <typename Medium, typename F> void for_each_view(Medium &m, F &&f) {
  f("Rkappa", m.h_memory_variable_kappa);
  f("Rxx", m.h_memory_variable_Rxx);
  f("Rxz", m.h_memory_variable_Rxz);
  f("EpsilonXX", m.h_epsilon_xx_att);
  f("EpsilonZZ", m.h_epsilon_zz_att);
  f("EpsilonXZ", m.h_epsilon_xz_att);
}

// Distinct function of the logical indices; `salt` separates datasets.
inline type_real pattern(const int salt, const std::size_t e,
                         const std::size_t iz, const std::size_t ix,
                         const std::size_t j) {
  return static_cast<type_real>(100000 * salt + 1000 * e + 100 * iz + 10 * ix +
                                j + 1);
}

// Applies f(salt, e, iz, ix, j, value_ref) over every logical entry.
template <typename View, typename F> void visit(View &view, F &&f) {
  constexpr std::size_t rank = View::rank();
  for (std::size_t e = 0; e < view.extent(0); ++e)
    for (std::size_t iz = 0; iz < view.extent(1); ++iz)
      for (std::size_t ix = 0; ix < view.extent(2); ++ix) {
        if constexpr (rank == 4) {
          for (std::size_t j = 0; j < view.extent(3); ++j)
            f(e, iz, ix, j, view(e, iz, ix, j));
        } else {
          f(e, iz, ix, std::size_t(0), view(e, iz, ix));
        }
      }
}

std::string attenuation_group_path(const std::string &dir,
                                   const specfem::element::medium_tag tag) {
  return dir + "/ForwardWavefield/Step000001/" +
         specfem::element::to_string(tag) + "/Attenuation";
}

// Fills every host view of every attenuation container with the pattern and
// pushes it to the device.
void fill_attenuation(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly) {
  specfem::tag_dispatch::for_each(
      AttenuationType::attenuation_medium_combinations,
      [&]<typename TagsType>() {
        auto &medium =
            assembly.attenuation.attenuation_storage.template get<TagsType>();
        if (medium.element_range.extent(0) == 0)
          return;
        int salt = 0;
        for_each_view(medium, [&](const char *, auto &view) {
          const int s = salt++;
          visit(view,
                [&](std::size_t e, std::size_t iz, std::size_t ix,
                    std::size_t j, auto &v) { v = pattern(s, e, iz, ix, j); });
        });
      });
  assembly.attenuation.copy_to_device();
}

void zero_attenuation(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly) {
  specfem::tag_dispatch::for_each(
      AttenuationType::attenuation_medium_combinations,
      [&]<typename TagsType>() {
        auto &medium =
            assembly.attenuation.attenuation_storage.template get<TagsType>();
        if (medium.element_range.extent(0) == 0)
          return;
        for_each_view(medium, [&](const char *, auto &view) {
          Kokkos::deep_copy(view.get_base_view(), type_real(0));
        });
      });
}

// Fills, writes one step, and zeroes the host views again.
void write_checkpoint(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const std::string &dir) {
  fill_attenuation(assembly);
  specfem::io::wavefield_writer<WriteBackend> writer(dir, true);
  writer.run(assembly, checkpoint_step);
  zero_attenuation(assembly);
}

} // namespace wavefield_checkpoint_impl

TEST_F(Assembly2D, AttenuationCheckpointRoundTrip) {
  namespace impl = wavefield_checkpoint_impl;
  int n_checked = 0;

  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);
    if (!Test.is_attenuation_enabled())
      continue;

    const std::string dir = impl::temp_directory(suffix + "_roundtrip");
    boost::filesystem::remove_all(dir);
    boost::filesystem::create_directories(dir);

    impl::write_checkpoint(assembly, dir);

    specfem::io::wavefield_reader<impl::ReadBackend> reader(dir);
    reader.run(assembly, impl::checkpoint_step);

    specfem::tag_dispatch::for_each(
        impl::AttenuationType::attenuation_medium_combinations,
        [&]<typename TagsType>() {
          auto &medium =
              assembly.attenuation.attenuation_storage.template get<TagsType>();
          const std::size_t nspec_attn = medium.element_range.extent(0);
          if (nspec_attn == 0)
            return;
          std::cout << "[wavefield_attenuation] " << Test.name << " / "
                    << specfem::element::to_string(TagsType::medium_tag)
                    << ": nspec_attn=" << nspec_attn << ", nspec_attn % "
                    << specfem::parallel_configuration::storage_chunk_size
                    << " = "
                    << nspec_attn %
                           specfem::parallel_configuration::storage_chunk_size
                    << std::endl;
          int salt = 0;
          impl::for_each_view(medium, [&](const char *name, auto &view) {
            const int s = salt++;
            std::size_t n_entries = 0;
            impl::visit(view, [&](std::size_t e, std::size_t iz, std::size_t ix,
                                  std::size_t j, auto &v) {
              EXPECT_NEAR(v, impl::pattern(s, e, iz, ix, j), 1e-3)
                  << Test.name << " " << name << " (" << e << "," << iz << ","
                  << ix << "," << j << ")";
              ++n_entries;
            });
            EXPECT_GT(n_entries, 0u) << name;
          });
          ++n_checked;
        });

    boost::filesystem::remove_all(dir);
  }
  EXPECT_GT(n_checked, 0) << "No attenuation container was exercised";
}

TEST_F(Assembly2D, AttenuationCheckpointHeaderIsRankOne) {
  namespace impl = wavefield_checkpoint_impl;
  int n_checked = 0;

  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);
    if (!Test.is_attenuation_enabled())
      continue;

    const std::string dir = impl::temp_directory(suffix + "_header");
    boost::filesystem::remove_all(dir);
    boost::filesystem::create_directories(dir);
    impl::write_checkpoint(assembly, dir);

    specfem::tag_dispatch::for_each(
        impl::AttenuationType::attenuation_medium_combinations,
        [&]<typename TagsType>() {
          auto &medium =
              assembly.attenuation.attenuation_storage.template get<TagsType>();
          if (medium.element_range.extent(0) == 0)
            return;
          const std::string group =
              impl::attenuation_group_path(dir, TagsType::medium_tag);

          const auto raw = specfem_test::layout_oracle::read_npy_raw<type_real>(
              group + "/Rxx.npy");
          ASSERT_EQ(raw.shape.size(), 1u);
          EXPECT_EQ(raw.data.size(),
                    medium.h_memory_variable_Rxx.get_base_view().extent(0));
          EXPECT_EQ(raw.shape[0],
                    medium.h_memory_variable_Rxx.get_base_view().extent(0));

          const auto stamp = specfem_test::layout_oracle::read_npy_raw<int>(
              group + "/storage_chunk_size.npy");
          ASSERT_EQ(stamp.data.size(), 1u);
          EXPECT_EQ(stamp.data[0],
                    specfem::parallel_configuration::storage_chunk_size);
          ++n_checked;
        });

    boost::filesystem::remove_all(dir);
  }
  EXPECT_GT(n_checked, 0) << "No attenuation container was exercised";
}

TEST_F(Assembly2D, AttenuationCheckpointChunkMismatchThrows) {
  namespace impl = wavefield_checkpoint_impl;
  int n_checked = 0;

  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);
    if (!Test.is_attenuation_enabled())
      continue;

    const std::string dir = impl::temp_directory(suffix + "_mismatch");
    boost::filesystem::remove_all(dir);
    boost::filesystem::create_directories(dir);
    impl::write_checkpoint(assembly, dir);

    // Overwrite the chunk-size stamp of every attenuating medium.
    const int n_before = n_checked;
    specfem::tag_dispatch::for_each(
        impl::AttenuationType::attenuation_medium_combinations,
        [&]<typename TagsType>() {
          auto &medium =
              assembly.attenuation.attenuation_storage.template get<TagsType>();
          if (medium.element_range.extent(0) == 0)
            return;
          Kokkos::View<int *, Kokkos::HostSpace> bad("bad_chunk_size", 1);
          bad(0) = specfem::parallel_configuration::storage_chunk_size + 1;
          boost::filesystem::path group_path(
              impl::attenuation_group_path(dir, TagsType::medium_tag));
          specfem::io_backends::impl::NPY::Dataset<decltype(bad),
                                                   specfem::io::write>(
              group_path, "storage_chunk_size", bad)
              .write();
          ++n_checked;
        });

    if (n_checked == n_before) {
      // No attenuating elements in this config: nothing is checkpointed.
      boost::filesystem::remove_all(dir);
      continue;
    }

    specfem::io::wavefield_reader<impl::ReadBackend> reader(dir);
    EXPECT_THROW(reader.run(assembly, impl::checkpoint_step),
                 std::runtime_error)
        << Test.name;

    boost::filesystem::remove_all(dir);
  }
  EXPECT_GT(n_checked, 0) << "No attenuation container was exercised";
}

TEST_F(Assembly2D, WavefieldCoordinatesLayout) {
  namespace impl = wavefield_checkpoint_impl;
  namespace oracle = specfem_test::layout_oracle;
  int n_media = 0;

  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);

    const std::string dir = impl::temp_directory(suffix + "_coordinates");
    boost::filesystem::remove_all(dir);
    boost::filesystem::create_directories(dir);

    {
      specfem::io::wavefield_writer<impl::WriteBackend> writer(dir, false);
      writer.initialize(assembly);
    }

    const int nspec = assembly.mesh.nspec;
    const int ngllz = assembly.mesh.element_grid.ngllz;
    const int ngllx = assembly.mesh.element_grid.ngllx;
    auto &forward = assembly.fields.forward;
    const int n_before = n_media;

    specfem::tag_dispatch::for_each(
        std::remove_reference_t<decltype(forward)>::combinations,
        [&]<typename TagsType>() {
          constexpr auto medium_tag = TagsType::medium_tag;
          const int nglob = forward.template get_nglob<medium_tag>();
          if (nglob == 0)
            return;
          const std::string group = dir + "/ForwardWavefield/Coordinates/" +
                                    specfem::element::to_string(medium_tag) +
                                    "/";
          const auto elements =
              assembly.element_types.get_elements_on_host(medium_tag);
          const std::size_t n_elements = elements.size();

          const auto x = oracle::read_npy_raw<type_real>(group + "X.npy");
          const auto z = oracle::read_npy_raw<type_real>(group + "Z.npy");
          const auto mapping = oracle::read_npy_raw<int>(group + "mapping.npy");
          const auto indices =
              oracle::read_npy_raw<int>(group + "element_indices.npy");

          ASSERT_EQ(x.shape.size(), 1u);
          ASSERT_EQ(z.shape.size(), 1u);
          EXPECT_EQ(x.data.size(), static_cast<std::size_t>(nglob));
          EXPECT_EQ(z.data.size(), static_cast<std::size_t>(nglob));
          ASSERT_EQ(mapping.shape.size(), 3u);
          EXPECT_EQ(mapping.shape[0], n_elements);
          EXPECT_EQ(mapping.shape[1], static_cast<std::size_t>(ngllz));
          EXPECT_EQ(mapping.shape[2], static_cast<std::size_t>(ngllx));
          EXPECT_FALSE(mapping.fortran_order);
          ASSERT_EQ(indices.shape.size(), 1u);
          ASSERT_EQ(indices.data.size(), n_elements);

          for (std::size_t e = 0; e < n_elements; ++e) {
            const int ispec = indices.data[e];
            ASSERT_GE(ispec, 0);
            ASSERT_LT(ispec, nspec);
            EXPECT_EQ(ispec, elements(e));
            for (int iz = 0; iz < ngllz; ++iz)
              for (int ix = 0; ix < ngllx; ++ix) {
                const int iglob = mapping.at({ e, static_cast<std::size_t>(iz),
                                               static_cast<std::size_t>(ix) });
                ASSERT_GE(iglob, 0);
                ASSERT_LT(iglob, nglob);
                EXPECT_NEAR(x.data[iglob],
                            assembly.mesh.h_coord(0, ispec, iz, ix), 1e-4)
                    << Test.name << " e=" << e << " iz=" << iz << " ix=" << ix;
                EXPECT_NEAR(z.data[iglob],
                            assembly.mesh.h_coord(1, ispec, iz, ix), 1e-4)
                    << Test.name << " e=" << e << " iz=" << iz << " ix=" << ix;
              }
          }
          std::cout << "[wavefield_coordinates] " << Test.name << " / "
                    << specfem::element::to_string(medium_tag)
                    << ": n_elements=" << n_elements << ", nglob=" << nglob
                    << std::endl;
          ++n_media;
        });

    boost::filesystem::remove_all(dir);
    if (n_media > n_before && n_media >= 2)
      break;
  }
  EXPECT_GT(n_media, 0) << "No medium was exercised";
}
