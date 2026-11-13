#include "../test_fixture/test_fixture.hpp"
#include "specfem/io.hpp"
#include "specfem/setup.hpp"
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

#ifndef NO_HDF5
#include "../../io/layout_oracle.hpp"
#endif

#include <cstddef>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#ifndef NO_HDF5
// Every heavy-data reference of Properties.xmf must name a dataset of
// Properties.h5 with the declared dimensions. The fluid-solid attenuation mesh
// gives one acoustic and one attenuating elastic group.
TEST_F(Assembly2D, PropertiesXdmfReferencesResolve) {
  int number_of_meshes_checked = 0;
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto assembly = std::get<5>(parameters);
    if (Test.name.find("Fluid-solid mesh with flat ocean bottom with "
                       "attenuation") == std::string::npos) {
      continue;
    }

    const std::string output_directory =
        (std::getenv("BUILD_DIR")
             ? std::string(std::getenv("BUILD_DIR"))
             : boost::filesystem::current_path().string()) +
        "/tests/unit-tests/temp_xdmf_io";
    boost::filesystem::create_directories(output_directory);

    specfem::io::property_writer<specfem::io_backends::HDF5<specfem::io::write>>
        writer(output_directory);
    writer.write(assembly);

    const std::string heavy_file = output_directory + "/Properties.h5";
    std::ifstream stream(output_directory + "/Properties.xmf");
    ASSERT_TRUE(stream.good()) << "Properties.xmf was not written";
    std::stringstream buffer;
    buffer << stream.rdbuf();
    const std::string xdmf = buffer.str();

    const std::regex reference(
        "<DataItem Dimensions=\"([0-9 ]+)\"[^>]*>Properties\\.h5:([^<]+)"
        "</DataItem>");
    int number_of_references = 0;
    for (auto match = std::sregex_iterator(xdmf.begin(), xdmf.end(), reference);
         match != std::sregex_iterator(); ++match) {
      std::vector<std::size_t> dimensions;
      std::istringstream dimension_stream((*match)[1].str());
      for (std::size_t extent; dimension_stream >> extent;) {
        dimensions.push_back(extent);
      }
      const std::string dataset_path = (*match)[2].str();
      const auto shape =
          (dataset_path.size() >= 13 &&
           dataset_path.substr(dataset_path.size() - 13) == "/connectivity")
              ? specfem_test::layout_oracle::read_hdf5_raw<int>(heavy_file,
                                                                dataset_path)
                    .shape
              : specfem_test::layout_oracle::read_hdf5_raw<type_real>(
                    heavy_file, dataset_path)
                    .shape;
      EXPECT_EQ(shape, dimensions) << dataset_path;
      ++number_of_references;
    }
    EXPECT_GT(number_of_references, 0);

    // One uniform grid per element group, each with a derived vp
    std::vector<std::string> grids;
    const std::string grid_marker = "GridType=\"Uniform\"";
    for (std::size_t start = xdmf.find(grid_marker);
         start != std::string::npos;) {
      const std::size_t next = xdmf.find(grid_marker, start + 1);
      grids.push_back(xdmf.substr(start, next - start));
      start = next;
    }
    ASSERT_EQ(grids.size(), 2);
    for (const auto &grid : grids) {
      EXPECT_NE(grid.find("<Attribute Name=\"vp\""), std::string::npos);
      EXPECT_NE(grid.find("ItemType=\"Function\""), std::string::npos);
    }
    EXPECT_NE(xdmf.find("/elastic_psv_isotropic_constant_isotropic/Qmu<"),
              std::string::npos);

    boost::filesystem::remove_all(output_directory);
    ++number_of_meshes_checked;
  }
  EXPECT_EQ(number_of_meshes_checked, 1);
}
#endif
