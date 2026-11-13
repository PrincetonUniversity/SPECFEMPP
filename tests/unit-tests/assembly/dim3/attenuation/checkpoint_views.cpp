#include "../../../io/layout_oracle.hpp"
#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/constants.hpp"
#include "specfem/datatype/domain_view.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/io.hpp"

#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <map>
#include <set>
#include <string>

namespace attenuation_checkpoint_views_test_impl {

using Medium = specfem::assembly::impl::attenuation_medium<
    specfem::element::dimension_tag::dim3,
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic,
    specfem::element::attenuation_tag::constant_isotropic>;

// Tiled storage has a padded tail whenever the storage chunk size does not
// divide this count.
constexpr int nspec = 5;
constexpr int ngll = 3;
constexpr int N_SLS = specfem::constants::N_SLS;

/** @brief A container holding only the checkpointed host views. */
Medium make_medium() {
  const auto vector_mirror = [](const std::string &name) {
    return specfem::datatype::create_mirror_view(
        Medium::view_type(name, nspec, ngll, ngll, ngll, N_SLS));
  };
  const auto scalar_mirror = [](const std::string &name) {
    return specfem::datatype::create_mirror_view(
        Medium::scalar_view_type(name, nspec, ngll, ngll, ngll));
  };

  Medium medium;
  medium.h_memory_variable_kappa = vector_mirror("Rkappa");
  medium.h_memory_variable_Rxx = vector_mirror("Rxx");
  medium.h_memory_variable_Ryy = vector_mirror("Ryy");
  medium.h_memory_variable_Rzz = vector_mirror("Rzz");
  medium.h_memory_variable_Rxy = vector_mirror("Rxy");
  medium.h_memory_variable_Rxz = vector_mirror("Rxz");
  medium.h_memory_variable_Ryz = vector_mirror("Ryz");
  medium.h_epsilon_xx_att = scalar_mirror("EpsilonXX");
  medium.h_epsilon_yy_att = scalar_mirror("EpsilonYY");
  medium.h_epsilon_zz_att = scalar_mirror("EpsilonZZ");
  medium.h_epsilon_xy_att = scalar_mirror("EpsilonXY");
  medium.h_epsilon_xz_att = scalar_mirror("EpsilonXZ");
  medium.h_epsilon_yz_att = scalar_mirror("EpsilonYZ");
  return medium;
}

/** @brief Distinct value per dataset and logical index. */
type_real pattern(const int salt, const int e, const int iz, const int iy,
                  const int ix, const int j) {
  return static_cast<type_real>(100000 * salt + 10000 * e + 1000 * iz +
                                100 * iy + 10 * ix + j + 1);
}

/** @brief Apply `f(e, iz, iy, ix, j, value)` to every logical entry. */
template <typename ViewType, typename FunctorType>
void visit(const ViewType &view, FunctorType f) {
  for (int e = 0; e < nspec; ++e)
    for (int iz = 0; iz < ngll; ++iz)
      for (int iy = 0; iy < ngll; ++iy)
        for (int ix = 0; ix < ngll; ++ix) {
          if constexpr (ViewType::rank() == 5) {
            for (int j = 0; j < N_SLS; ++j)
              f(e, iz, iy, ix, j, view(e, iz, iy, ix, j));
          } else {
            f(e, iz, iy, ix, 0, view(e, iz, iy, ix));
          }
        }
}

} // namespace attenuation_checkpoint_views_test_impl

TEST(AttenuationCheckpointViews, VisitsEveryDatasetOnce) {
  namespace impl = attenuation_checkpoint_views_test_impl;
  const impl::Medium medium = impl::make_medium();

  std::map<std::string, int> visits;
  medium.for_each_checkpoint_view(
      [&](const std::string &name, const auto &) { ++visits[name]; });

  const std::set<std::string> expected = {
    "Rkappa",    "Rxx",       "Ryy",       "Rzz",       "Rxy",
    "Rxz",       "Ryz",       "EpsilonXX", "EpsilonYY", "EpsilonZZ",
    "EpsilonXY", "EpsilonXZ", "EpsilonYZ"
  };
  EXPECT_EQ(visits.size(), expected.size());
  for (const auto &name : expected)
    EXPECT_EQ(visits[name], 1) << name;
}

TEST(AttenuationCheckpointViews, BaseViewsRoundTripThroughNpy) {
  namespace impl = attenuation_checkpoint_views_test_impl;
  const impl::Medium medium = impl::make_medium();

  const std::string dir =
      (std::getenv("BUILD_DIR") ? std::string(std::getenv("BUILD_DIR"))
                                : boost::filesystem::current_path().string()) +
      "/tests/unit-tests/temp_attenuation_checkpoint_views";
  boost::filesystem::remove_all(dir);

  int salt = 0;
  medium.for_each_checkpoint_view([&](const std::string &, const auto &view) {
    const int s = salt++;
    impl::visit(view, [&](int e, int iz, int iy, int ix, int j, auto &v) {
      v = impl::pattern(s, e, iz, iy, ix, j);
    });
  });

  {
    specfem::io_backends::NPY<specfem::io::write>::File file(dir);
    medium.for_each_checkpoint_view([&](const std::string &name,
                                        const auto &view) {
      file.createDataset(name, specfem::datatype::storage_view(view)).write();
    });
    file.flush();
  }

  // The file holds the rank-1 storage exactly as it is in memory.
  medium.for_each_checkpoint_view(
      [&](const std::string &name, const auto &view) {
        const auto base = specfem::datatype::storage_view(view);
        const auto raw = specfem_test::layout_oracle::read_npy_raw<type_real>(
            dir + "/" + name + ".npy");
        ASSERT_EQ(raw.shape.size(), 1u) << name;
        ASSERT_EQ(raw.shape[0], base.extent(0)) << name;
        for (std::size_t i = 0; i < base.extent(0); ++i)
          EXPECT_NEAR(raw.data[i], base(i), 1e-3) << name << " i=" << i;
        Kokkos::deep_copy(base, type_real(0));
      });

  {
    specfem::io_backends::NPY<specfem::io::read>::File file(dir);
    medium.for_each_checkpoint_view(
        [&](const std::string &name, const auto &view) {
          file.openDataset(name, specfem::datatype::storage_view(view)).read();
        });
  }

  salt = 0;
  medium.for_each_checkpoint_view(
      [&](const std::string &name, const auto &view) {
        const int s = salt++;
        impl::visit(view, [&](int e, int iz, int iy, int ix, int j, auto &v) {
          EXPECT_NEAR(v, impl::pattern(s, e, iz, iy, ix, j), 1e-3)
              << name << " (" << e << "," << iz << "," << iy << "," << ix << ","
              << j << ")";
        });
      });

  boost::filesystem::remove_all(dir);
}
