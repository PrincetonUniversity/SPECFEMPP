#include "specfem/io/xdmf/writer.hpp"
#include "specfem/setup.hpp"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

template <specfem::element::dimension_tag DimensionTag>
void specfem::io::xdmf::write_element_groups(
    const boost::filesystem::path &xdmf_path,
    const std::string &heavy_file_name,
    const std::vector<specfem::io::xdmf::ElementGroupDescription<DimensionTag>>
        &groups) {

  constexpr bool is_dim2 =
      (DimensionTag == specfem::element::dimension_tag::dim2);
  constexpr int nodes_per_cell = is_dim2 ? 4 : 8;

  std::ofstream stream(xdmf_path.string());
  if (!stream) {
    throw std::runtime_error("XDMF writer: cannot open " + xdmf_path.string());
  }

  const auto dimensions = [](const std::vector<std::size_t> &extents) {
    std::string text;
    for (const std::size_t extent : extents) {
      text += (text.empty() ? "" : " ") + std::to_string(extent);
    }
    return text;
  };

  const auto data_item =
      [&](const std::string &indent, const std::vector<std::size_t> &extents,
          const std::string &number_type, const std::size_t precision,
          const std::string &heavy_path) {
        stream << indent << "<DataItem Dimensions=\"" << dimensions(extents)
               << "\" NumberType=\"" << number_type << "\" Precision=\""
               << precision << "\" Format=\"HDF\">" << heavy_file_name << ":"
               << heavy_path << "</DataItem>\n";
      };

  const std::string collection_name =
      boost::filesystem::path(heavy_file_name).stem().string();

  stream << "<?xml version=\"1.0\" ?>\n"
         << "<Xdmf Version=\"2.0\">\n"
         << "  <Domain>\n"
         << "    <Grid Name=\"" << collection_name
         << "\" GridType=\"Collection\" CollectionType=\"Spatial\">\n";

  for (const auto &group : groups) {
    const std::size_t name_start = group.group.find_first_not_of('/');
    const std::string grid_name = (name_start == std::string::npos)
                                      ? std::string()
                                      : group.group.substr(name_start);
    const auto point_extents = specfem::io::xdmf::element_extents(
        group.number_of_elements, group.element_grid);
    std::size_t number_of_cells = point_extents[0];
    for (std::size_t i = 1; i < point_extents.size(); ++i) {
      number_of_cells *= point_extents[i] - 1;
    }

    stream << "      <Grid Name=\"" << grid_name
           << "\" GridType=\"Uniform\">\n";

    stream << "        <Topology TopologyType=\""
           << (is_dim2 ? "Quadrilateral" : "Hexahedron")
           << "\" NumberOfElements=\"" << number_of_cells << "\">\n";
    data_item("          ",
              { number_of_cells, static_cast<std::size_t>(nodes_per_cell) },
              "Int", sizeof(int), group.group + "/connectivity");
    stream << "        </Topology>\n";

    stream << "        <Geometry GeometryType=\"" << (is_dim2 ? "X_Y" : "X_Y_Z")
           << "\">\n";
    const std::vector<std::string> coordinates =
        is_dim2 ? std::vector<std::string>{ "X", "Z" }
                : std::vector<std::string>{ "X", "Y", "Z" };
    for (const auto &coordinate : coordinates) {
      data_item("          ", point_extents, "Float", sizeof(type_real),
                group.group + "/" + coordinate);
    }
    stream << "        </Geometry>\n";

    for (const auto &dataset : group.datasets) {
      stream << "        <Attribute Name=\"" << dataset.name
             << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
      data_item("          ", dataset.extents, "Float", sizeof(type_real),
                group.group + "/" + dataset.name);
      stream << "        </Attribute>\n";
    }

    for (const auto &derived : group.derived) {
      std::vector<std::vector<std::size_t>> operand_extents;
      for (const auto &operand : derived.operands) {
        const auto stored = std::find_if(
            group.datasets.begin(), group.datasets.end(),
            [&](const auto &dataset) { return dataset.name == operand; });
        if (stored == group.datasets.end()) {
          throw std::runtime_error("XDMF writer: derived quantity " +
                                   derived.name + " of group " + group.group +
                                   " uses " + operand +
                                   ", which is not a stored dataset");
        }
        operand_extents.push_back(stored->extents);
      }
      if (operand_extents.empty()) {
        throw std::runtime_error("XDMF writer: derived quantity " +
                                 derived.name + " has no operands");
      }

      stream << "        <Attribute Name=\"" << derived.name
             << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
             << "          <DataItem ItemType=\"Function\" Function=\""
             << derived.function << "\" Dimensions=\""
             << dimensions(operand_extents.front()) << "\">\n";
      for (std::size_t i = 0; i < derived.operands.size(); ++i) {
        data_item("            ", operand_extents[i], "Float",
                  sizeof(type_real), group.group + "/" + derived.operands[i]);
      }
      stream << "          </DataItem>\n"
             << "        </Attribute>\n";
    }

    stream << "      </Grid>\n";
  }

  stream << "    </Grid>\n"
         << "  </Domain>\n"
         << "</Xdmf>\n";
}

template void
specfem::io::xdmf::write_element_groups<specfem::element::dimension_tag::dim2>(
    const boost::filesystem::path &, const std::string &,
    const std::vector<specfem::io::xdmf::ElementGroupDescription<
        specfem::element::dimension_tag::dim2>> &);

template void
specfem::io::xdmf::write_element_groups<specfem::element::dimension_tag::dim3>(
    const boost::filesystem::path &, const std::string &,
    const std::vector<specfem::io::xdmf::ElementGroupDescription<
        specfem::element::dimension_tag::dim3>> &);
