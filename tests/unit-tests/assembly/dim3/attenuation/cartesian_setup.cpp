#include "specfem/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/io/mesh/impl/fortran/dim3/read_materials.hpp"
#include "specfem/medium_container.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"

#include "globe_model_evaluator.h"
#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace cartesian_attenuation_setup_test_impl {

constexpr auto dimension = specfem::element::dimension_tag::dim3;
using Medium = specfem::element::medium_tag;
using Property = specfem::element::property_tag;
using AttenuationTag = specfem::element::attenuation_tag;
using Assembly = specfem::assembly::assembly<dimension>;

constexpr type_real rho = 2300.0;
constexpr type_real vp = 2800.0;
constexpr type_real vs = 1500.0;
constexpr type_real Qkappa = 500.0;
constexpr type_real Qmu = 80.0;

/** @brief Stiffnesses of the material assigned to every element. */
specfem::io::mesh::impl::fortran::dim3_impl::stiffness_array stiffnesses() {
  return specfem::io::mesh::impl::fortran::dim3_impl::model_aniso(1, rho, vp,
                                                                  vs);
}

/**
 * @brief A small Cartesian mesh whose elements all share one anisotropic
 *        elastic material.
 *
 * @param attenuating Enable attenuation and give the material Qkappa/Qmu;
 * otherwise attenuation is disabled and the material is elastic
 */
specfem::mesh::cartesian3d_mesh make_mesh(const bool attenuating) {
  using specfem::units::unit_symbols::Hz;
  const specfem::attenuation::Setup setup{
    attenuating, 1.0 * Hz,
    specfem::utilities::Band<specfem::units::Hertz>(0.1 * Hz, 10.0 * Hz)
  };
  auto mesh = specfem::io::read_3d_mesh(
      "data/dim3/EightNodeElastic/database.bin", setup);

  const auto c = stiffnesses();
  const auto attenuation_tag =
      attenuating ? AttenuationTag::constant_isotropic : AttenuationTag::none;
  specfem::mesh::materials<dimension> materials;
  int index = 0;
  if (attenuating) {
    index = materials.add_material(
        specfem::medium_container::material<dimension, Medium::elastic,
                                            Property::anisotropic,
                                            AttenuationTag::constant_isotropic>(
            rho, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
            c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18],
            c[19], c[20], Qkappa, Qmu));
  } else {
    index = materials.add_material(specfem::medium_container::material<
                                   dimension, Medium::elastic,
                                   Property::anisotropic, AttenuationTag::none>(
        rho, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10],
        c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18], c[19], c[20]));
  }
  materials.nspec = mesh.nspec;
  materials.material_index_mapping.assign(
      mesh.nspec,
      { Medium::elastic, Property::anisotropic, attenuation_tag, index, 0 });
  mesh.materials = materials;
  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    auto &tags = mesh.tags.tags_container(ispec);
    tags.medium_tag = Medium::elastic;
    tags.property_tag = Property::anisotropic;
    tags.attenuation_tag = attenuation_tag;
  }
  return mesh;
}

/** @brief Assemble @p mesh without sources or receivers. */
std::unique_ptr<Assembly>
make_assembly(const specfem::mesh::cartesian3d_mesh &mesh,
              const std::shared_ptr<specfem::io::reader> &property_reader) {
  std::vector<std::shared_ptr<specfem::sources::source<dimension>>> sources;
  const std::vector<std::shared_ptr<specfem::receivers::receiver<dimension>>>
      receivers;
  const specfem::quadrature::quadratures quadrature(
      specfem::quadrature::gll::gll{});
  return std::make_unique<Assembly>(
      mesh, quadrature, sources, receivers,
      std::vector<specfem::enums::wavefield>{
          specfem::enums::wavefield::displacement },
      0.0, 1.0e-3, 10, 10, 1, specfem::simulation::type::forward, false,
      property_reader);
}

/** @brief Host views of an anisotropic property container, c11 ... c66. */
template <typename Container> auto stiffness_views(const Container &container) {
  std::vector<std::decay_t<decltype(container.h_c11)>> views;
  container.for_each_host_view([&](const auto view, const std::string &name) {
    if (name != "rho")
      views.push_back(view);
  });
  return views;
}

/** @brief Globe's independent Cartesian (north-pole) attenuation of cij. */
struct Expected {
  double cij[21];
  double mu_unrelaxed;
  double rates[specfem::constants::N_SLS];

  Expected(const Assembly &assembly, const double qmu) {
    const auto c = stiffnesses();
    double original[21];
    for (int k = 0; k < 21; ++k)
      original[k] = c[k];
    // At the north pole the globe's radial frame is the Cartesian frame.
    globe_evaluator_attenuate_cij(
        0., 0., assembly.attenuation.band.min.raw(),
        assembly.attenuation.band.max.raw(), qmu, assembly.attenuation.fc.raw(),
        assembly.attenuation.f0.raw(), original, cij, &mu_unrelaxed, rates);
  }
};

constexpr double tolerance = std::is_same_v<type_real, double> ? 1.e-10 : 2.e-6;
constexpr double rate_tolerance =
    std::is_same_v<type_real, double> ? 1.e-10 : 2.e-3;
// ASCII stores 10 significant digits; recompute re-fits the read Q.
constexpr double io_tolerance =
    std::is_same_v<type_real, double> ? 1.e-8 : 2.e-6;
constexpr double io_rate_tolerance =
    std::is_same_v<type_real, double> ? 1.e-6 : 2.e-3;

} // namespace cartesian_attenuation_setup_test_impl

TEST(CartesianAnisotropicAttenuationSetup, MaterialsAreShiftedOnAssembly) {
  namespace test = cartesian_attenuation_setup_test_impl;
  const auto mesh = test::make_mesh(true);
  const auto assembly = test::make_assembly(mesh, nullptr);
  const auto &props =
      assembly->properties
          .get_container<test::Medium::elastic, test::Property::anisotropic>();
  const auto &att =
      assembly->attenuation
          .get_container<test::Medium::elastic, test::Property::anisotropic>();
  ASSERT_EQ(att.element_range.size(), mesh.nspec);
  ASSERT_EQ(props.element_range.size(), mesh.nspec);

  const test::Expected expected(*assembly, test::Qmu);
  const auto views = test::stiffness_views(props);
  ASSERT_EQ(views.size(), 21);

  // The shifted properties must also have reached the device.
  std::vector<std::decay_t<decltype(props.h_c11)>> device_copies;
  props.for_each_device_view([&](const auto view, const std::string &name) {
    if (name == "rho")
      return;
    auto copy = specfem::datatype::create_mirror_view(view);
    specfem::datatype::deep_copy(copy, view);
    device_copies.push_back(copy);
  });
  ASSERT_EQ(device_copies.size(), 21);

  const auto &grid = assembly->mesh.element_grid;
  for (int i = 0; i < mesh.nspec; ++i)
    for (int z = 0; z < grid.ngllz; ++z)
      for (int y = 0; y < grid.nglly; ++y)
        for (int x = 0; x < grid.ngllx; ++x) {
          for (int k = 0; k < 21; ++k) {
            EXPECT_NEAR(views[k](i, z, y, x), expected.cij[k],
                        test::tolerance * expected.cij[0])
                << "component " << k;
            EXPECT_NEAR(device_copies[k](i, z, y, x), views[k](i, z, y, x), 0.)
                << "component " << k;
          }
          EXPECT_NEAR(props.h_rho(i, z, y, x), test::rho, 0.);
          EXPECT_NEAR(att.h_Qkappa(i, z, y, x), test::Qkappa, 0.);
          EXPECT_NEAR(att.h_Qmu(i, z, y, x), test::Qmu, 0.);
          EXPECT_NEAR(att.h_kappa_scale(i, z, y, x), 1., 0.);
          for (int j = 0; j < specfem::constants::N_SLS; ++j) {
            EXPECT_NEAR(att.h_mu_relaxation_rate(i, z, y, x, j),
                        expected.rates[j],
                        test::rate_tolerance * std::abs(expected.rates[j]));
            EXPECT_NEAR(att.h_kappa_relaxation_rate(i, z, y, x, j), 0., 0.);
          }
        }
}

TEST(CartesianAnisotropicAttenuationSetup, DisabledAttenuationKeepsMaterials) {
  namespace test = cartesian_attenuation_setup_test_impl;
  const auto mesh = test::make_mesh(false);
  const auto assembly = test::make_assembly(mesh, nullptr);
  const auto &props =
      assembly->properties
          .get_container<test::Medium::elastic, test::Property::anisotropic>();
  const auto &att =
      assembly->attenuation
          .get_container<test::Medium::elastic, test::Property::anisotropic>();
  EXPECT_EQ(att.element_range.size(), 0);
  const auto c = test::stiffnesses();
  const auto views = test::stiffness_views(props);
  ASSERT_EQ(views.size(), 21);
  const auto &grid = assembly->mesh.element_grid;
  for (int i = 0; i < mesh.nspec; ++i)
    for (int z = 0; z < grid.ngllz; ++z)
      for (int y = 0; y < grid.nglly; ++y)
        for (int x = 0; x < grid.ngllx; ++x)
          for (int k = 0; k < 21; ++k)
            EXPECT_NEAR(views[k](i, z, y, x), c[k], 0.) << "component " << k;
}

TEST(CartesianAnisotropicAttenuationSetup, PropertyReadUsesCartesianContext) {
  namespace test = cartesian_attenuation_setup_test_impl;
  const auto mesh = test::make_mesh(true);
  const auto source = test::make_assembly(mesh, nullptr);
  ASSERT_GT(mesh.nspec, 0);
  // The read must shift in the Cartesian frame: a globe context would need
  // the spherical coordinates, which Cartesian meshes do not have.

  const std::string folder =
      (std::getenv("BUILD_DIR") ? std::string(std::getenv("BUILD_DIR"))
                                : boost::filesystem::current_path().string()) +
      "/tests/unit-tests/temp_cartesian_anisotropic_attenuation_io";
  boost::filesystem::create_directories(folder);
  specfem::io::property_writer<specfem::io_backends::ASCII<specfem::io::write>>(
      folder)
      .write(*source);

  const auto reader = std::make_shared<specfem::io::property_reader<
      specfem::io_backends::ASCII<specfem::io::read>>>(folder);
  const auto assembly = test::make_assembly(mesh, reader);
  boost::filesystem::remove_all(folder);

  const auto &props =
      assembly->properties
          .get_container<test::Medium::elastic, test::Property::anisotropic>();
  const auto &att =
      assembly->attenuation
          .get_container<test::Medium::elastic, test::Property::anisotropic>();
  const test::Expected expected(*assembly, test::Qmu);
  const auto c = test::stiffnesses();
  const auto views = test::stiffness_views(props);
  ASSERT_EQ(views.size(), 21);

  const auto &grid = assembly->mesh.element_grid;
  for (int i = 0; i < mesh.nspec; ++i)
    for (int z = 0; z < grid.ngllz; ++z)
      for (int y = 0; y < grid.nglly; ++y)
        for (int x = 0; x < grid.ngllx; ++x) {
          for (int k = 0; k < 21; ++k) {
            EXPECT_NEAR(views[k](i, z, y, x), expected.cij[k],
                        test::io_tolerance * c[0])
                << "element " << i << ", component " << k;
          }
          EXPECT_NEAR(att.h_Qkappa(i, z, y, x), test::Qkappa,
                      test::io_tolerance * test::Qkappa);
          EXPECT_NEAR(att.h_Qmu(i, z, y, x), test::Qmu,
                      test::io_tolerance * test::Qmu);
          EXPECT_NEAR(att.h_kappa_scale(i, z, y, x), 1., 0.);
          for (int j = 0; j < specfem::constants::N_SLS; ++j) {
            EXPECT_NEAR(att.h_mu_relaxation_rate(i, z, y, x, j),
                        expected.rates[j],
                        test::io_rate_tolerance * std::abs(expected.rates[j]))
                << "element " << i;
            EXPECT_NEAR(att.h_kappa_relaxation_rate(i, z, y, x, j), 0., 0.);
          }
        }
}
