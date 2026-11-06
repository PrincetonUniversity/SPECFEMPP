#include "SPECFEM_Environment.hpp"
#include "specfem/assembly/impl/value_containers.hpp"
#include "specfem/element.hpp"
#include "specfem/io/xdmf/derived_quantities.hpp"
#include "specfem/io/xdmf/description.hpp"
#include "specfem/io/xdmf/writer.hpp"
#include "specfem/medium_container/domain_properties.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

#include <cctype>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem_test::xdmf {

/// Collapse whitespace runs to one space and drop whitespace between tags
std::string normalize_whitespace(const std::string &text) {
  std::string collapsed;
  bool in_whitespace = false;
  for (const char character : text) {
    if (std::isspace(static_cast<unsigned char>(character))) {
      in_whitespace = true;
      continue;
    }
    if (in_whitespace && !collapsed.empty())
      collapsed += ' ';
    in_whitespace = false;
    collapsed += character;
  }
  return std::regex_replace(collapsed, std::regex("> <"), "><");
}

/// Replace every "@P@" with the byte size of type_real
std::string with_float_precision(const std::string &text) {
  return std::regex_replace(text, std::regex("@P@"),
                            std::to_string(sizeof(type_real)));
}

std::string read_file(const boost::filesystem::path &path) {
  std::ifstream stream(path.string());
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

} // namespace specfem_test::xdmf

TEST(XdmfWriter, TwoGroups2D) {
  using DimensionTag = specfem::element::dimension_tag;
  const boost::filesystem::path path = "xdmf_writer_two_groups_2d.xmf";

  specfem::io::xdmf::ElementGroupDescription<DimensionTag::dim2> acoustic;
  acoustic.group = "/acoustic_isotropic_none";
  acoustic.number_of_elements = 2;
  acoustic.element_grid =
      specfem::mesh_entity::element_grid<DimensionTag::dim2>(3);
  acoustic.datasets = { { "rho_inverse", { 2, 3, 3 } },
                        { "kappa", { 2, 3, 3 } } };
  acoustic.derived = { { "vp", "sqrt($0 * $1)", { "kappa", "rho_inverse" } },
                       { "rho", "$0 / ($0 * $0)", { "rho_inverse" } } };

  specfem::io::xdmf::ElementGroupDescription<DimensionTag::dim2> elastic;
  elastic.group = "/elastic_psv_isotropic_constant_isotropic";
  elastic.number_of_elements = 1;
  elastic.element_grid =
      specfem::mesh_entity::element_grid<DimensionTag::dim2>(3);
  elastic.datasets = { { "kappa", { 1, 3, 3 } },
                       { "mu", { 1, 3, 3 } },
                       { "rho", { 1, 3, 3 } },
                       { "Qkappa", { 1, 3, 3 } },
                       { "Qmu", { 1, 3, 3 } } };

  specfem::io::xdmf::write_element_groups<DimensionTag::dim2>(
      path, "Properties.h5", { acoustic, elastic });

  const std::string expected = R"xml(
<?xml version="1.0" ?>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="Properties" GridType="Collection" CollectionType="Spatial">
      <Grid Name="acoustic_isotropic_none" GridType="Uniform">
        <Topology TopologyType="Quadrilateral" NumberOfElements="8">
          <DataItem Dimensions="8 4" NumberType="Int" Precision="4" Format="HDF">Properties.h5:/acoustic_isotropic_none/connectivity</DataItem>
        </Topology>
        <Geometry GeometryType="X_Y">
          <DataItem Dimensions="2 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/acoustic_isotropic_none/X</DataItem>
          <DataItem Dimensions="2 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/acoustic_isotropic_none/Z</DataItem>
        </Geometry>
        <Attribute Name="rho_inverse" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="2 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/acoustic_isotropic_none/rho_inverse</DataItem>
        </Attribute>
        <Attribute Name="kappa" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="2 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/acoustic_isotropic_none/kappa</DataItem>
        </Attribute>
        <Attribute Name="vp" AttributeType="Scalar" Center="Node">
          <DataItem ItemType="Function" Function="sqrt($0 * $1)" Dimensions="2 3 3">
            <DataItem Dimensions="2 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/acoustic_isotropic_none/kappa</DataItem>
            <DataItem Dimensions="2 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/acoustic_isotropic_none/rho_inverse</DataItem>
          </DataItem>
        </Attribute>
        <Attribute Name="rho" AttributeType="Scalar" Center="Node">
          <DataItem ItemType="Function" Function="$0 / ($0 * $0)" Dimensions="2 3 3">
            <DataItem Dimensions="2 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/acoustic_isotropic_none/rho_inverse</DataItem>
          </DataItem>
        </Attribute>
      </Grid>
      <Grid Name="elastic_psv_isotropic_constant_isotropic" GridType="Uniform">
        <Topology TopologyType="Quadrilateral" NumberOfElements="4">
          <DataItem Dimensions="4 4" NumberType="Int" Precision="4" Format="HDF">Properties.h5:/elastic_psv_isotropic_constant_isotropic/connectivity</DataItem>
        </Topology>
        <Geometry GeometryType="X_Y">
          <DataItem Dimensions="1 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/elastic_psv_isotropic_constant_isotropic/X</DataItem>
          <DataItem Dimensions="1 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/elastic_psv_isotropic_constant_isotropic/Z</DataItem>
        </Geometry>
        <Attribute Name="kappa" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="1 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/elastic_psv_isotropic_constant_isotropic/kappa</DataItem>
        </Attribute>
        <Attribute Name="mu" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="1 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/elastic_psv_isotropic_constant_isotropic/mu</DataItem>
        </Attribute>
        <Attribute Name="rho" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="1 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/elastic_psv_isotropic_constant_isotropic/rho</DataItem>
        </Attribute>
        <Attribute Name="Qkappa" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="1 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/elastic_psv_isotropic_constant_isotropic/Qkappa</DataItem>
        </Attribute>
        <Attribute Name="Qmu" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="1 3 3" NumberType="Float" Precision="@P@" Format="HDF">Properties.h5:/elastic_psv_isotropic_constant_isotropic/Qmu</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
)xml";

  EXPECT_EQ(specfem_test::xdmf::normalize_whitespace(
                specfem_test::xdmf::read_file(path)),
            specfem_test::xdmf::normalize_whitespace(
                specfem_test::xdmf::with_float_precision(expected)));
  boost::filesystem::remove(path);
}

TEST(XdmfWriter, OneGroup3D) {
  using DimensionTag = specfem::element::dimension_tag;
  const boost::filesystem::path path = "xdmf_writer_one_group_3d.xmf";

  specfem::io::xdmf::ElementGroupDescription<DimensionTag::dim3> elastic;
  elastic.group = "/elastic_isotropic";
  elastic.number_of_elements = 1;
  elastic.element_grid =
      specfem::mesh_entity::element_grid<DimensionTag::dim3>(2, 2, 2);
  elastic.datasets = { { "mu", { 1, 2, 2, 2 } }, { "rho", { 1, 2, 2, 2 } } };
  elastic.derived = { { "vs", "sqrt($0 / $1)", { "mu", "rho" } } };

  specfem::io::xdmf::write_element_groups<DimensionTag::dim3>(
      path, "Kernels.h5", { elastic });

  const std::string expected = R"xml(
<?xml version="1.0" ?>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="Kernels" GridType="Collection" CollectionType="Spatial">
      <Grid Name="elastic_isotropic" GridType="Uniform">
        <Topology TopologyType="Hexahedron" NumberOfElements="1">
          <DataItem Dimensions="1 8" NumberType="Int" Precision="4" Format="HDF">Kernels.h5:/elastic_isotropic/connectivity</DataItem>
        </Topology>
        <Geometry GeometryType="X_Y_Z">
          <DataItem Dimensions="1 2 2 2" NumberType="Float" Precision="@P@" Format="HDF">Kernels.h5:/elastic_isotropic/X</DataItem>
          <DataItem Dimensions="1 2 2 2" NumberType="Float" Precision="@P@" Format="HDF">Kernels.h5:/elastic_isotropic/Y</DataItem>
          <DataItem Dimensions="1 2 2 2" NumberType="Float" Precision="@P@" Format="HDF">Kernels.h5:/elastic_isotropic/Z</DataItem>
        </Geometry>
        <Attribute Name="mu" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="1 2 2 2" NumberType="Float" Precision="@P@" Format="HDF">Kernels.h5:/elastic_isotropic/mu</DataItem>
        </Attribute>
        <Attribute Name="rho" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="1 2 2 2" NumberType="Float" Precision="@P@" Format="HDF">Kernels.h5:/elastic_isotropic/rho</DataItem>
        </Attribute>
        <Attribute Name="vs" AttributeType="Scalar" Center="Node">
          <DataItem ItemType="Function" Function="sqrt($0 / $1)" Dimensions="1 2 2 2">
            <DataItem Dimensions="1 2 2 2" NumberType="Float" Precision="@P@" Format="HDF">Kernels.h5:/elastic_isotropic/mu</DataItem>
            <DataItem Dimensions="1 2 2 2" NumberType="Float" Precision="@P@" Format="HDF">Kernels.h5:/elastic_isotropic/rho</DataItem>
          </DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
)xml";

  EXPECT_EQ(specfem_test::xdmf::normalize_whitespace(
                specfem_test::xdmf::read_file(path)),
            specfem_test::xdmf::normalize_whitespace(
                specfem_test::xdmf::with_float_precision(expected)));
  boost::filesystem::remove(path);
}

TEST(XdmfWriter, RejectsOperandThatIsNotStored) {
  using DimensionTag = specfem::element::dimension_tag;
  const boost::filesystem::path path = "xdmf_writer_missing_operand.xmf";

  specfem::io::xdmf::ElementGroupDescription<DimensionTag::dim2> group;
  group.group = "/elastic_psv_isotropic_none";
  group.number_of_elements = 1;
  group.element_grid =
      specfem::mesh_entity::element_grid<DimensionTag::dim2>(2);
  group.datasets = { { "mu", { 1, 2, 2 } } };
  group.derived = { { "vs", "sqrt($0 / $1)", { "mu", "rho" } } };

  EXPECT_THROW(specfem::io::xdmf::write_element_groups<DimensionTag::dim2>(
                   path, "Properties.h5", { group }),
               std::runtime_error);
  boost::filesystem::remove(path);
}

/**
 * @brief Check the derived table of every (medium, property) combination of
 *        @p DimensionTag against the stored property names.
 */
template <specfem::element::dimension_tag DimensionTag>
void check_derived_quantities() {
  int number_of_combinations = 0;
  specfem::tag_dispatch::for_each(
      specfem::assembly::impl::ContainerSets<DimensionTag>::combinations,
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;
        const std::string combination =
            specfem::element::to_string(medium_tag, property_tag);

        const specfem::medium_container::properties::data_container<
            DimensionTag, medium_tag, property_tag>
            container(1, specfem::mesh_entity::element_grid<DimensionTag>(2));
        std::set<std::string> stored;
        container.for_each_host_view(
            [&](const auto &, const std::string &name) {
              stored.insert(name);
            });

        const auto derived =
            specfem::io::xdmf::DerivedQuantities<medium_tag,
                                                 property_tag>::list();
        for (const auto &quantity : derived) {
          EXPECT_FALSE(quantity.operands.empty())
              << combination << " / " << quantity.name;
          for (const auto &operand : quantity.operands) {
            EXPECT_TRUE(stored.count(operand))
                << combination << " / " << quantity.name << ": " << operand
                << " is not a stored property";
          }
          const std::regex operand_reference("\\$([0-9]+)");
          for (auto match = std::sregex_iterator(quantity.function.begin(),
                                                 quantity.function.end(),
                                                 operand_reference);
               match != std::sregex_iterator(); ++match) {
            EXPECT_LT(std::stoul((*match)[1].str()), quantity.operands.size())
                << combination << " / " << quantity.name;
          }
          // Xdmf2 applies only lower-case function names and evaluates a
          // numeric literal left of '-' or '/' as the right operand.
          EXPECT_EQ(quantity.function.find("SQRT"), std::string::npos)
              << combination << " / " << quantity.name;
          EXPECT_FALSE(std::regex_search(
              quantity.function,
              std::regex("(^|[^$0-9.])[0-9]+(\\.[0-9]*)?\\s*[-/]")))
              << combination << " / " << quantity.name << ": "
              << quantity.function;
          // Longer expressions failed to evaluate in vtkXdmfReader
          EXPECT_LE(quantity.function.size(), 140)
              << combination << " / " << quantity.name;
          EXPECT_EQ(stored.count(quantity.name), 0)
              << combination << " / " << quantity.name
              << " shadows a stored property";
        }
        ++number_of_combinations;
      });
  EXPECT_GT(number_of_combinations, 0);
}

TEST(XdmfDerivedQuantities, OperandsAreStoredProperties2D) {
  check_derived_quantities<specfem::element::dimension_tag::dim2>();
}

TEST(XdmfDerivedQuantities, OperandsAreStoredProperties3D) {
  check_derived_quantities<specfem::element::dimension_tag::dim3>();
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
