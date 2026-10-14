#include "specfem/io.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
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

// Attenuation period band written to every database, in seconds.
constexpr double min_attenuation_period = 20.0;
constexpr double max_attenuation_period = 1000.0;

/** @brief Source frequency consistent with the written period band. */
double consistent_source_frequency() {
  return 1.0 / std::sqrt(min_attenuation_period * max_attenuation_period);
}

/** @brief Default model flags: only HONOR_1D_SPHERICAL_MOHO is set. */
std::vector<int> default_model_flags() {
  std::vector<int> flags(16, 0);
  flags[11] = 1;
  return flags;
}

/** @brief Contents of the synthetic single-element database. */
struct DatabaseOptions {
  bool attenuation = false;
  double source_frequency = 0.0;
  int region_code = 1; ///< 1 = crust/mantle, 2 = outer core
  int medium_code = 2; ///< 1 = acoustic, 2 = elastic
  int property_tag = 0;
  bool include_mpi = false;
  bool has_reference_geometry = false;
  double planet_radius = 6371000.0;
  std::vector<int> model_flags = default_model_flags();
};

std::filesystem::path write_database(const DatabaseOptions &options = {}) {
  const bool attenuation = options.attenuation;
  const bool include_mpi = options.include_mpi;
  const bool has_reference_geometry = options.has_reference_geometry;
  const double planet_radius = options.planet_radius;
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path =
      std::filesystem::temp_directory_path() /
      ("specfempp_globe_reader_" + std::to_string(suffix) + ".bin");
  std::ofstream stream(path, std::ios::binary);

  Record header;
  header.append_fixed("SPECFEMPP_GLOBE_DB", 32);
  header.append(5);
  header.write(stream);

  const std::vector<double> planet_values = {
    planet_radius, 5514.3, (1.0 - 1.0 / 299.8) * (1.0 - 1.0 / 299.8),
    24.0,          3600.0, 9000.0,
  };
  write_values(stream, 1, 2, static_cast<int>(planet_values.size()));
  write_values(stream, planet_values);
  write_values(stream, 27, 5, 5, 5, 1);
  write_values(stream, 0, 0, 0, 0, 0, attenuation ? 1 : 0, 0,
               has_reference_geometry ? 1 : 0);
  write_values(stream, 1);

  Record model;
  model.append_fixed("1D_isotropic_prem", 512);
  model.write(stream);
  write_values(stream, 5, std::vector<int>{ 1, 0, 0, 2, 0 });
  write_values(stream, static_cast<int>(options.model_flags.size()),
               options.model_flags);
  write_values(stream, 6, 8, 8);
  write_values(stream, min_attenuation_period, max_attenuation_period,
               options.source_frequency);

  write_values(stream, 27);
  std::vector<double> x(27), y(27), z(27);
  for (int inode = 0; inode < 27; ++inode) {
    x[inode] = 1000.0 + inode;
    y[inode] = 2000.0 + inode;
    z[inode] = 3000.0 + inode;
  }
  write_values(stream, x, y, z);
  if (has_reference_geometry) {
    std::vector<double> x_ref(27), y_ref(27), z_ref(27);
    for (int inode = 0; inode < 27; ++inode) {
      x_ref[inode] = 10000.0 + inode;
      y_ref[inode] = 20000.0 + inode;
      z_ref[inode] = 30000.0 + inode;
    }
    write_values(stream, x_ref, y_ref, z_ref);
  }

  write_values(stream, 1);
  write_values(stream, std::vector<int>{ options.region_code },
               std::vector<int>{ options.medium_code },
               std::vector<int>{ options.property_tag }, std::vector<int>{ 4 });
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
  write_values(stream, include_mpi ? 1 : 0);
  if (include_mpi) {
    write_values(stream, 1, 1, 1, 1, 3, 19, 23);
  }
  stream.close();
  return path;
}

} // namespace globe_reader_test_impl

TEST(GlobeMeshReader, ReadsThinDatabaseAndPreservesReferenceContext) {
  const auto path = globe_reader_test_impl::write_database();
  const auto mesh = specfem::io::read_globe_mesh(path.string(),
                                                 specfem::attenuation::Setup{});
  std::filesystem::remove(path);

  EXPECT_EQ(mesh.nspec, 1);
  EXPECT_EQ(mesh.control_nodes.ngnod, 27);
  EXPECT_EQ(mesh.control_nodes.nnodes, 27);
  EXPECT_EQ(mesh.globe.model_config.model_name, "1D_isotropic_prem");
  EXPECT_EQ(mesh.globe.model_verification.codes,
            (std::vector<int>{ 1, 0, 0, 2, 0 }));
  std::vector<bool> expected_flags(16, false);
  expected_flags[11] = true;
  EXPECT_EQ(mesh.globe.model_verification.flags, expected_flags);
  EXPECT_EQ(mesh.globe.model_config.nchunks, 6);
  ASSERT_EQ(mesh.globe.element_context.size(), 1);
  EXPECT_EQ(mesh.globe.element_context[0].region,
            specfem::element::region_tag::crust_mantle);
  EXPECT_EQ(mesh.globe.element_context[0].idoubling, 4);
  EXPECT_FALSE(mesh.globe.element_context[0].element_in_crust);
  EXPECT_TRUE(mesh.globe.element_context[0].element_in_mantle);
  EXPECT_DOUBLE_EQ(mesh.globe.reference_coordinates(26, 2), 3026.0);
  EXPECT_EQ(mesh.control_nodes.control_node_index(0, 26), 26);
  EXPECT_EQ(mesh.boundaries.acoustic_free_surface.nelem_acoustic_surface, 1);
}

TEST(GlobeMeshReader, RejectsInvalidPlanetConstants) {
  const auto path =
      globe_reader_test_impl::write_database({ .planet_radius = -1.0 });
  EXPECT_THROW(specfem::io::read_globe_mesh(path.string(),
                                            specfem::attenuation::Setup{}),
               std::runtime_error);
  std::filesystem::remove(path);
}

TEST(GlobeMeshReader, ReadsSeparateReferenceGeometry) {
  const auto path = globe_reader_test_impl::write_database(
      { .has_reference_geometry = true });
  const auto mesh = specfem::io::read_globe_mesh(path.string(),
                                                 specfem::attenuation::Setup{});
  std::filesystem::remove(path);

  EXPECT_TRUE(mesh.globe.has_reference_geometry);
  EXPECT_DOUBLE_EQ(mesh.control_nodes.coordinates(26, 2), 3026.0);
  EXPECT_DOUBLE_EQ(mesh.globe.reference_coordinates(26, 0), 10026.0);
  EXPECT_DOUBLE_EQ(mesh.globe.reference_coordinates(26, 1), 20026.0);
  EXPECT_DOUBLE_EQ(mesh.globe.reference_coordinates(26, 2), 30026.0);
}

TEST(GlobeMeshReader, RejectsAnInconsistentAttenuationSourceFrequency) {
  const auto path = globe_reader_test_impl::write_database(
      { .attenuation = true, .source_frequency = 1.0 });
  EXPECT_THROW(specfem::io::read_globe_mesh(path.string(),
                                            specfem::attenuation::Setup{}),
               std::runtime_error);
  std::filesystem::remove(path);
}

TEST(GlobeMeshReader, PreservesAnisotropicElasticPropertyTag) {
  const auto path =
      globe_reader_test_impl::write_database({ .property_tag = 1 });
  const auto mesh = specfem::io::read_globe_mesh(path.string(),
                                                 specfem::attenuation::Setup{});
  std::filesystem::remove(path);

  const auto &mapping = mesh.materials.material_index_mapping.front();
  EXPECT_EQ(mapping.type, specfem::element::medium_tag::elastic);
  EXPECT_EQ(mapping.property, specfem::element::property_tag::anisotropic);
  EXPECT_EQ(mapping.attenuation, specfem::element::attenuation_tag::none);
}

TEST(GlobeMeshReader, ReadsResolvedMpiAdjacency) {
  const auto path =
      globe_reader_test_impl::write_database({ .include_mpi = true });
  const auto mesh = specfem::io::read_globe_mesh(path.string(),
                                                 specfem::attenuation::Setup{});
  std::filesystem::remove(path);

  const auto &connections = mesh.adjacency_graph.mpi_connections();
  ASSERT_EQ(connections.size(), 1);
  EXPECT_EQ(connections[0].orientation,
            specfem::mesh_entity::dim3::type::bottom);
  EXPECT_EQ(connections[0].neighbor_partition, 1);
  EXPECT_EQ(connections[0].neighbor_orientation,
            specfem::mesh_entity::dim3::type::top);
  EXPECT_EQ(connections[0].local_index, 0);
  EXPECT_EQ(connections[0].neighbor_local_index, 0);
  EXPECT_EQ(connections[0].local_anchor_point,
            specfem::mesh_entity::dim3::type::bottom_front_left);
  EXPECT_EQ(connections[0].neighbor_anchor_point,
            specfem::mesh_entity::dim3::type::top_front_left);
}

namespace globe_reader_test_impl {

/** @brief Runtime attenuation setup; the band defaults to the database's. */
specfem::attenuation::Setup
attenuation_setup(const double min_frequency = 1.0 / max_attenuation_period,
                  const double max_frequency = 1.0 / min_attenuation_period) {
  using specfem::units::unit_symbols::Hz;
  return { true, 1.0 * Hz,
           specfem::utilities::Band<specfem::units::Hertz>(
               min_frequency * Hz, max_frequency * Hz) };
}

/** @brief Database options for an attenuating single-element mesh. */
DatabaseOptions attenuating_database() {
  return { .attenuation = true,
           .source_frequency = consistent_source_frequency() };
}

/** @brief Read a written database and remove it, also when reading throws. */
specfem::mesh::globe3d_mesh
read_and_remove(const std::filesystem::path &path,
                const specfem::attenuation::Setup &setup) {
  struct Remove {
    const std::filesystem::path &path;
    ~Remove() { std::filesystem::remove(path); }
  } remove{ path };
  return specfem::io::read_globe_mesh(path.string(), setup);
}

} // namespace globe_reader_test_impl

TEST(GlobeMeshReader, TagsElasticElementsFromTheAttenuationFlag) {
  namespace test = globe_reader_test_impl;
  const auto mesh =
      test::read_and_remove(test::write_database(test::attenuating_database()),
                            test::attenuation_setup());

  const auto &mapping = mesh.materials.material_index_mapping.front();
  EXPECT_EQ(mapping.type, specfem::element::medium_tag::elastic);
  EXPECT_EQ(mapping.attenuation,
            specfem::element::attenuation_tag::constant_isotropic);
  EXPECT_TRUE(mesh.attenuation.enabled);
  EXPECT_FALSE(mesh.attenuation.bulk_attenuation);
  // The band is held in type_real, which may be single precision.
  EXPECT_NEAR(mesh.attenuation.band.min.raw(), 1.0 / 1000.0, 1.0e-6 / 1000.0);
  EXPECT_NEAR(mesh.attenuation.band.max.raw(), 1.0 / 20.0, 1.0e-6 / 20.0);
}

TEST(GlobeMeshReader, LeavesAcousticElementsNonAttenuating) {
  namespace test = globe_reader_test_impl;
  auto options = test::attenuating_database();
  options.region_code = 2;
  options.medium_code = 1;
  const auto mesh = test::read_and_remove(test::write_database(options),
                                          test::attenuation_setup());

  const auto &mapping = mesh.materials.material_index_mapping.front();
  EXPECT_EQ(mapping.type, specfem::element::medium_tag::acoustic);
  EXPECT_EQ(mapping.attenuation, specfem::element::attenuation_tag::none);
}

TEST(GlobeMeshReader, RejectsAttenuationForADatabaseMeshedWithoutIt) {
  namespace test = globe_reader_test_impl;
  EXPECT_THROW(
      test::read_and_remove(test::write_database(), test::attenuation_setup()),
      std::runtime_error);
}

TEST(GlobeMeshReader, RunsElasticWhenAttenuationIsDisabledAtRuntime) {
  namespace test = globe_reader_test_impl;
  const auto mesh =
      test::read_and_remove(test::write_database(test::attenuating_database()),
                            specfem::attenuation::Setup{});

  const auto &mapping = mesh.materials.material_index_mapping.front();
  EXPECT_EQ(mapping.attenuation, specfem::element::attenuation_tag::none);
  EXPECT_FALSE(mesh.attenuation.enabled);
}

TEST(GlobeMeshReader, RejectsABandThatDiffersFromTheDatabase) {
  namespace test = globe_reader_test_impl;
  EXPECT_THROW(
      test::read_and_remove(test::write_database(test::attenuating_database()),
                            test::attenuation_setup(1.0 / 1000.0, 1.0 / 10.0)),
      std::runtime_error);
  EXPECT_THROW(
      test::read_and_remove(test::write_database(test::attenuating_database()),
                            test::attenuation_setup(1.0 / 2000.0, 1.0 / 20.0)),
      std::runtime_error);
}

TEST(GlobeMeshReader, RejectsAttenuationWithoutAReferenceFrequency) {
  namespace test = globe_reader_test_impl;
  auto setup = test::attenuation_setup();
  setup.f0.reset();
  EXPECT_THROW(test::read_and_remove(
                   test::write_database(test::attenuating_database()), setup),
               std::runtime_error);
}

TEST(GlobeMeshReader, RejectsUnvalidatedAttenuationModels) {
  namespace test = globe_reader_test_impl;
  // ATTENUATION_3D_BERKELEY and ATTENUATION_GLL model-flag positions.
  for (const int flag : { 9, 10 }) {
    SCOPED_TRACE("model flag " + std::to_string(flag));
    auto options = test::attenuating_database();
    options.model_flags[flag] = 1;
    EXPECT_THROW(test::read_and_remove(test::write_database(options),
                                       test::attenuation_setup()),
                 std::runtime_error);

    // Without attenuation the flag is irrelevant.
    auto elastic_options = test::DatabaseOptions{};
    elastic_options.model_flags[flag] = 1;
    EXPECT_NO_THROW(test::read_and_remove(test::write_database(elastic_options),
                                          specfem::attenuation::Setup{}));
  }
}
