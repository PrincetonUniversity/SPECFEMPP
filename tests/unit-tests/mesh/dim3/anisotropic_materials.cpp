#include "specfem/enums.hpp"
#include "specfem/io/mesh/impl/fortran/dim3/read_materials.hpp"
#include "specfem/medium_container.hpp"
#include "specfem/mesh.hpp"

#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace anisotropic_materials_test_impl {

constexpr int ngnod = 8;
constexpr type_real rho = 2300.0;
constexpr type_real vp = 2800.0;
constexpr type_real vs = 1500.0;
constexpr int iflag_aniso = 1;

/** @brief Append one Fortran unformatted sequential record. */
template <typename T>
void write_record(std::ofstream &stream, const std::vector<T> &values) {
  const std::int32_t length =
      static_cast<std::int32_t>(values.size() * sizeof(T));
  stream.write(reinterpret_cast<const char *>(&length), sizeof(length));
  stream.write(reinterpret_cast<const char *>(values.data()), length);
  stream.write(reinterpret_cast<const char *>(&length), sizeof(length));
}

/**
 * @brief Run read_materials on a one-element MESHFEM3D database holding a
 *        single anisotropic elastic material.
 *
 * @param Qkappa Bulk quality factor stored in the database
 * @param Qmu Shear quality factor stored in the database
 * @param attenuation_enabled Whether the simulation enables attenuation
 * @return Materials read back from the database
 */
specfem::mesh::materials<specfem::element::dimension_tag::dim3>
read_anisotropic_material(const double Qkappa, const double Qmu,
                          const bool attenuation_enabled) {
  const std::string path =
      (boost::filesystem::temp_directory_path() /
       boost::filesystem::unique_path("anisotropic_materials_%%%%%%%%.bin"))
          .string();
  {
    std::ofstream stream(path, std::ios::binary);
    write_record<std::int32_t>(stream, { 1, 0 });
    std::vector<double> properties(17, 0.0);
    properties[0] = rho;
    properties[1] = vp;
    properties[2] = vs;
    properties[3] = Qkappa;
    properties[4] = Qmu;
    properties[5] = iflag_aniso;
    properties[6] = 2; // elastic domain
    write_record<double>(stream, properties);
    write_record<std::int32_t>(stream, { 1 });       // nspec
    write_record<std::int32_t>(stream, { 5, 5, 5 }); // ngllz, nglly, ngllx
    std::vector<std::int32_t> element = { 1, 1, 0 };
    for (int inode = 1; inode <= ngnod; ++inode)
      element.push_back(inode);
    write_record<std::int32_t>(stream, element);
  }
  std::ifstream stream(path, std::ios::binary);
  auto result = specfem::io::mesh::impl::fortran::dim3::read_materials(
      stream, ngnod, attenuation_enabled);
  stream.close();
  boost::filesystem::remove(path);
  return std::get<5>(result);
}

/** @brief Check the stored stiffnesses and density against model_aniso. */
template <typename PointProperties>
void expect_model_aniso(const PointProperties &properties) {
  const auto c = specfem::io::mesh::impl::fortran::dim3_impl::model_aniso(
      iflag_aniso, rho, vp, vs);
  const type_real actual[21] = {
    properties.c11(), properties.c12(), properties.c13(), properties.c14(),
    properties.c15(), properties.c16(), properties.c22(), properties.c23(),
    properties.c24(), properties.c25(), properties.c26(), properties.c33(),
    properties.c34(), properties.c35(), properties.c36(), properties.c44(),
    properties.c45(), properties.c46(), properties.c55(), properties.c56(),
    properties.c66()
  };
  for (int k = 0; k < 21; ++k)
    EXPECT_NEAR(actual[k], c[k], 0.) << "component " << k;
  EXPECT_NEAR(properties.rho(), rho, 0.);
}

} // namespace anisotropic_materials_test_impl

TEST(AnisotropicMaterials, AttenuatingMaterialKeepsQualityFactors) {
  namespace test = anisotropic_materials_test_impl;
  const auto materials = test::read_anisotropic_material(500.0, 80.0, true);
  ASSERT_EQ(materials.material_index_mapping.size(), 1);
  const auto &mapping = materials.material_index_mapping[0];
  EXPECT_EQ(mapping.type, specfem::element::medium_tag::elastic);
  EXPECT_EQ(mapping.property, specfem::element::property_tag::anisotropic);
  EXPECT_EQ(mapping.attenuation,
            specfem::element::attenuation_tag::constant_isotropic);

  const auto material =
      materials
          .get_material<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::anisotropic,
                        specfem::element::attenuation_tag::constant_isotropic>(
              0);
  EXPECT_NEAR(material.Qkappa, 500.0, 0.);
  EXPECT_NEAR(material.Qmu, 80.0, 0.);
  test::expect_model_aniso(material.get_properties());
}

TEST(AnisotropicMaterials, SingleSentinelStillAttenuates) {
  namespace test = anisotropic_materials_test_impl;
  // Only Qkappa = Qmu = 9999 flags no attenuation; Qkappa = 9999 alone does
  // not.
  const auto materials = test::read_anisotropic_material(9999.0, 80.0, true);
  ASSERT_EQ(materials.material_index_mapping.size(), 1);
  EXPECT_EQ(materials.material_index_mapping[0].attenuation,
            specfem::element::attenuation_tag::constant_isotropic);
  const auto material =
      materials
          .get_material<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::anisotropic,
                        specfem::element::attenuation_tag::constant_isotropic>(
              0);
  EXPECT_NEAR(material.Qkappa, 9999.0, 0.);
  EXPECT_NEAR(material.Qmu, 80.0, 0.);
}

TEST(AnisotropicMaterials, PairedSentinelOrDisabledIsElastic) {
  namespace test = anisotropic_materials_test_impl;
  struct Case {
    double Qkappa;
    double Qmu;
    bool attenuation_enabled;
  };
  for (const auto &[Qkappa, Qmu, enabled] :
       { Case{ 9999.0, 9999.0, true }, Case{ 500.0, 80.0, false } }) {
    const auto materials =
        test::read_anisotropic_material(Qkappa, Qmu, enabled);
    ASSERT_EQ(materials.material_index_mapping.size(), 1);
    const auto &mapping = materials.material_index_mapping[0];
    EXPECT_EQ(mapping.property, specfem::element::property_tag::anisotropic);
    EXPECT_EQ(mapping.attenuation, specfem::element::attenuation_tag::none)
        << "Qkappa = " << Qkappa << ", Qmu = " << Qmu
        << ", enabled = " << enabled;
    const auto material =
        materials.get_material<specfem::element::medium_tag::elastic,
                               specfem::element::property_tag::anisotropic,
                               specfem::element::attenuation_tag::none>(0);
    test::expect_model_aniso(material.get_properties());
  }
}

TEST(AnisotropicMaterials, ConstructorStoresQualityFactors) {
  using Material = specfem::medium_container::material<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic,
      specfem::element::attenuation_tag::constant_isotropic>;
  const Material material(3000.0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
                          12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
                          400.0, 70.0);
  const auto properties = material.get_properties();
  EXPECT_NEAR(properties.rho(), 3000.0, 0.);
  EXPECT_NEAR(properties.c11(), 1., 0.);
  EXPECT_NEAR(properties.c44(), 16., 0.);
  EXPECT_NEAR(properties.c66(), 21., 0.);
  EXPECT_NEAR(material.Qkappa, 400.0, 0.);
  EXPECT_NEAR(material.Qmu, 70.0, 0.);
  // Without quality factors (globe placeholders) both default to 9999.
  const Material placeholder(3000.0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
                             11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
                             21.);
  EXPECT_NEAR(placeholder.Qkappa, 9999.0, 0.);
  EXPECT_NEAR(placeholder.Qmu, 9999.0, 0.);
}
