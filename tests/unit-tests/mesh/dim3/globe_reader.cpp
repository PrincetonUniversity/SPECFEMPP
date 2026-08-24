#include "specfem/io.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <vector>

namespace globe_reader_test_impl {

class Record {
public:
  template <typename T> void append(const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto *bytes = reinterpret_cast<const char *>(&value);
    data.insert(data.end(), bytes, bytes + sizeof(T));
  }

  template <typename T> void append(const std::vector<T> &values) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto *bytes = reinterpret_cast<const char *>(values.data());
    data.insert(data.end(), bytes, bytes + values.size() * sizeof(T));
  }

  void append_fixed(const std::string &value, const std::size_t size) {
    const auto old_size = data.size();
    data.resize(old_size + size, ' ');
    std::memcpy(data.data() + old_size, value.data(),
                std::min(value.size(), size));
  }

  void write(std::ofstream &stream) const {
    const int size = static_cast<int>(data.size());
    stream.write(reinterpret_cast<const char *>(&size), sizeof(size));
    stream.write(data.data(), size);
    stream.write(reinterpret_cast<const char *>(&size), sizeof(size));
  }

private:
  std::vector<char> data;
};

template <typename... Values>
void write_values(std::ofstream &stream, const Values &...values) {
  Record record;
  (record.append(values), ...);
  record.write(stream);
}

void write_surface(std::ofstream &stream, const std::vector<int> &elements,
                   const std::vector<int> &faces) {
  write_values(stream, static_cast<int>(elements.size()));
  if (!elements.empty()) {
    write_values(stream, elements, faces);
  }
}

std::filesystem::path write_database(const bool attenuation = false,
                                     const double source_frequency = 0.0) {
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path =
      std::filesystem::temp_directory_path() /
      ("specfempp_globe_reader_" + std::to_string(suffix) + ".bin");
  std::ofstream stream(path, std::ios::binary);

  Record header;
  header.append_fixed("SPECFEMPP_GLOBE_DB", 32);
  header.append(2);
  header.write(stream);

  write_values(stream, 1, 6371000.0, 5514.3);
  write_values(stream, 27, 5, 5, 5, 1);
  write_values(stream, 0, 0, 0, 0, 0, attenuation ? 1 : 0, 0, 0);
  write_values(stream, 1);

  Record model;
  model.append_fixed("PREM", 512);
  model.write(stream);
  write_values(stream, 5, std::vector<int>{ 1, 0, 0, 0, 0 });
  write_values(stream, 16, std::vector<int>(16, 0));
  write_values(stream, 6, 8, 8);
  write_values(stream, 20.0, 1000.0, source_frequency);

  write_values(stream, 27);
  std::vector<double> x(27), y(27), z(27);
  for (int inode = 0; inode < 27; ++inode) {
    x[inode] = 1000.0 + inode;
    y[inode] = 2000.0 + inode;
    z[inode] = 3000.0 + inode;
  }
  write_values(stream, x, y, z);

  write_values(stream, 1);
  write_values(stream, std::vector<int>{ 1 }, std::vector<int>{ 2 },
               std::vector<int>{ 0 }, std::vector<int>{ 4 });
  write_values(stream, std::vector<double>{ 3000000.0 },
               std::vector<double>{ 3100000.0 });
  write_values(stream, std::vector<int>{ 0 }, std::vector<int>{ 1 });
  std::vector<int> node_ids(27);
  for (int inode = 0; inode < 27; ++inode) {
    node_ids[inode] = inode + 1;
  }
  write_values(stream, node_ids);

  write_surface(stream, { 1 }, { 3 });
  write_surface(stream, {}, {});
  write_surface(stream, {}, {});
  write_surface(stream, {}, {});

  write_values(stream, 0);
  write_values(stream, std::vector<int>{ 1, 1 });
  write_values(stream, std::vector<int>{});
  write_values(stream, std::vector<int>{});
  write_values(stream, 0);
  stream.close();
  return path;
}

} // namespace globe_reader_test_impl

TEST(GlobeMeshReader, ReadsThinDatabaseAndPreservesReferenceContext) {
  const auto path = globe_reader_test_impl::write_database();
  const auto mesh = specfem::io::read_globe_mesh(path.string(),
                                                 specfem::attenuation::Setup{});
  std::filesystem::remove(path);

  ASSERT_TRUE(mesh.globe.has_value());
  EXPECT_EQ(mesh.nspec, 1);
  EXPECT_EQ(mesh.control_nodes.ngnod, 27);
  EXPECT_EQ(mesh.control_nodes.nnodes, 27);
  EXPECT_EQ(mesh.globe->model_config.model_name, "PREM");
  EXPECT_EQ(mesh.globe->model_verification.codes,
            (std::vector<int>{ 1, 0, 0, 0, 0 }));
  EXPECT_EQ(mesh.globe->model_config.nchunks, 6);
  ASSERT_EQ(mesh.globe->element_context.size(), 1);
  EXPECT_EQ(mesh.globe->element_context[0].region, 1);
  EXPECT_EQ(mesh.globe->element_context[0].idoubling, 4);
  EXPECT_FALSE(mesh.globe->element_context[0].element_in_crust);
  EXPECT_TRUE(mesh.globe->element_context[0].element_in_mantle);
  EXPECT_DOUBLE_EQ(mesh.globe->reference_coordinates(26, 2), 3026.0);
  EXPECT_EQ(mesh.control_nodes.control_node_index(0, 26), 26);
  EXPECT_EQ(mesh.boundaries.acoustic_free_surface.nelem_acoustic_surface, 1);
}

TEST(GlobeMeshReader, RejectsAnInconsistentAttenuationSourceFrequency) {
  const auto path = globe_reader_test_impl::write_database(true, 1.0);
  EXPECT_THROW(specfem::io::read_globe_mesh(path.string(),
                                            specfem::attenuation::Setup{}),
               std::runtime_error);
  std::filesystem::remove(path);
}
