#include "specfem/io.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <utility>
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

/** @brief Write a surface given as (one-based element, face code) pairs. */
void write_surface(std::ofstream &stream,
                   const std::vector<std::pair<int, int>> &entries) {
  std::vector<int> elements;
  std::vector<int> faces;
  for (const auto &[element, face] : entries) {
    elements.push_back(element);
    faces.push_back(face);
  }
  write_surface(stream, elements, faces);
}

/**
 * @brief Contents of a synthetic globe database.
 *
 * Elements form a radial column, bottom to top: element k's top face touches
 * element k + 1's bottom face.
 */
struct DatabaseOptions {
  bool attenuation = false;
  double source_frequency = 0.0;
  int property_tag = 0;
  bool include_mpi = false;
  bool has_reference_geometry = false;
  double planet_radius = 6371000.0;
  /// One globe region code per element, bottom to top (2 = outer core).
  std::vector<int> region_codes = { 1 };
  /// CMB entries as (one-based element, face code).
  std::vector<std::pair<int, int>> cmb_faces = {};
  int nchunks = 6;
  bool oceans = false;
  /// Ocean-load entries as (one-based element, face code).
  std::vector<std::pair<int, int>> ocean_faces = {};
};

std::filesystem::path write_database(const DatabaseOptions &options = {}) {
  constexpr int face_bottom = 1;
  constexpr int face_top = 3;
  const int nspec = static_cast<int>(options.region_codes.size());

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
    options.planet_radius,
    5514.3,
    (1.0 - 1.0 / 299.8) * (1.0 - 1.0 / 299.8),
    24.0,
    3600.0,
    9000.0,
  };
  write_values(stream, 1, 2, static_cast<int>(planet_values.size()));
  write_values(stream, planet_values);
  write_values(stream, 27, 5, 5, 5, 1);
  write_values(stream, 0, 0, 0, 0, 0, options.attenuation ? 1 : 0,
               options.oceans ? 1 : 0, options.has_reference_geometry ? 1 : 0);
  write_values(stream, 1);

  Record model;
  model.append_fixed("1D_isotropic_prem", 512);
  model.write(stream);
  write_values(stream, 5, std::vector<int>{ 1, 0, 0, 0, 0 });
  write_values(stream, 16, std::vector<int>(16, 0));
  write_values(stream, options.nchunks, 8, 8);
  write_values(stream, 20.0, 1000.0, options.source_frequency);

  write_values(stream, 27);
  std::vector<double> x(27), y(27), z(27);
  for (int inode = 0; inode < 27; ++inode) {
    x[inode] = 1000.0 + inode;
    y[inode] = 2000.0 + inode;
    z[inode] = 3000.0 + inode;
  }
  write_values(stream, x, y, z);
  if (options.has_reference_geometry) {
    std::vector<double> x_ref(27), y_ref(27), z_ref(27);
    for (int inode = 0; inode < 27; ++inode) {
      x_ref[inode] = 10000.0 + inode;
      y_ref[inode] = 20000.0 + inode;
      z_ref[inode] = 30000.0 + inode;
    }
    write_values(stream, x_ref, y_ref, z_ref);
  }

  write_values(stream, nspec);
  // Region 2 is the fluid outer core, so it carries the acoustic medium code;
  // every other region is solid and elastic.
  std::vector<int> medium_codes(nspec);
  std::transform(options.region_codes.begin(), options.region_codes.end(),
                 medium_codes.begin(),
                 [](const int region) { return (region == 2) ? 1 : 2; });
  write_values(stream, options.region_codes, medium_codes,
               std::vector<int>(nspec, options.property_tag),
               std::vector<int>(nspec, 4));
  write_values(stream, std::vector<double>(nspec, 3000000.0),
               std::vector<double>(nspec, 3100000.0));
  write_values(stream, std::vector<int>(nspec, 0), std::vector<int>(nspec, 1));
  // Every element reuses the same 27 anchors. The reader bounds-checks node
  // ids but never inspects geometry, and the face checks need none.
  std::vector<int> node_ids(27 * nspec);
  for (int inode = 0; inode < 27 * nspec; ++inode) {
    node_ids[inode] = (inode % 27) + 1;
  }
  write_values(stream, node_ids);

  write_surface(stream, { nspec }, { face_top });
  write_surface(stream, options.cmb_faces);
  write_surface(stream, {}, {});
  write_surface(stream, options.ocean_faces);

  // One-based CSR adjacency of the radial column.
  std::vector<int> xadj = { 1 };
  std::vector<int> adjncy;
  std::vector<int> adjacency_types;
  for (int ispec = 1; ispec <= nspec; ++ispec) {
    if (ispec > 1) {
      adjncy.push_back(ispec - 1);
      adjacency_types.push_back(face_bottom);
    }
    if (ispec < nspec) {
      adjncy.push_back(ispec + 1);
      adjacency_types.push_back(face_top);
    }
    xadj.push_back(static_cast<int>(adjncy.size()) + 1);
  }
  write_values(stream, static_cast<int>(adjncy.size()));
  write_values(stream, xadj);
  write_values(stream, adjncy);
  write_values(stream, adjacency_types);
  write_values(stream, options.include_mpi ? 1 : 0);
  if (options.include_mpi) {
    write_values(stream, 1, 1, 1, 1, 3, 19, 23);
  }
  stream.close();
  return path;
}

/**
 * @brief Expect reading @p path to throw a @c std::runtime_error whose message
 * contains @p reason, so that a failure for an unrelated cause does not pass.
 */
void expect_read_throws_with(const std::filesystem::path &path,
                             const std::string &reason) {
  try {
    specfem::io::read_globe_mesh(path.string(), specfem::attenuation::Setup{});
    ADD_FAILURE() << "expected a throw containing \"" << reason << "\"";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(reason), std::string::npos)
        << "expected \"" << reason << "\" in: " << error.what();
  }
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
            (std::vector<int>{ 1, 0, 0, 0, 0 }));
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

// The CMB and ICB are assembled from local connections only, so a fluid-solid
// face split across ranks would be dropped from the coupling with no error.
// check_supported rejects the only locally detectable form of that: an
// outer-core element owning a radial face on an MPI boundary. A solid element
// with the same connection is fine -- that is the case above.
TEST(GlobeMeshReader, RejectsRadialMpiFaceOnFluidElement) {
  const auto path = globe_reader_test_impl::write_database(
      { .include_mpi = true, .region_codes = { 2 } });
  globe_reader_test_impl::expect_read_throws_with(path,
                                                  "Unsupported globe mesh");
  std::filesystem::remove(path);
}

// A single element has no neighbor, so medium contrast implies no interface
// faces. Recording one as CMB makes the database contradict itself, which
// check_consistency must catch.
TEST(GlobeMeshReader, RejectsCmbFaceWithoutMediumContrast) {
  const auto path =
      globe_reader_test_impl::write_database({ .cmb_faces = { { 1, 3 } } });
  globe_reader_test_impl::expect_read_throws_with(path,
                                                  "Recorded but not implied");
  std::filesystem::remove(path);
}

// The fluid-solid tests below use a two-element column: outer core (element 1)
// below crust/mantle (element 2), so their shared face is a CMB face. The
// mesher records both sides of it: the outer-core top and the mantle bottom.

TEST(GlobeMeshReader, AcceptsCmbRecordedOnBothSides) {
  const auto path = globe_reader_test_impl::write_database(
      { .region_codes = { 2, 1 }, .cmb_faces = { { 1, 3 }, { 2, 1 } } });
  EXPECT_NO_THROW(specfem::io::read_globe_mesh(path.string(),
                                               specfem::attenuation::Setup{}));
  std::filesystem::remove(path);
}

TEST(GlobeMeshReader, RejectsFluidSolidFaceMissingFromCmb) {
  const auto path =
      globe_reader_test_impl::write_database({ .region_codes = { 2, 1 } });
  globe_reader_test_impl::expect_read_throws_with(path,
                                                  "Implied but not recorded");
  std::filesystem::remove(path);
}

TEST(GlobeMeshReader, RejectsCmbRecordedOnOneSideOnly) {
  const auto path = globe_reader_test_impl::write_database(
      { .region_codes = { 2, 1 }, .cmb_faces = { { 1, 3 } } });
  globe_reader_test_impl::expect_read_throws_with(path,
                                                  "Implied but not recorded");
  std::filesystem::remove(path);
}

TEST(GlobeMeshReader, RejectsUnmeshableChunkCount) {
  const auto path = globe_reader_test_impl::write_database({ .nchunks = 4 });
  globe_reader_test_impl::expect_read_throws_with(path, "1, 2, 3 or 6");
  std::filesystem::remove(path);
}

// The free surface of the single-element database is the top face (code 3) of
// element 1, so these place the ocean load on or off it.

TEST(GlobeMeshReader, AcceptsOceanLoadOnFreeSurface) {
  const auto path = globe_reader_test_impl::write_database(
      { .oceans = true, .ocean_faces = { { 1, 3 } } });
  EXPECT_NO_THROW(specfem::io::read_globe_mesh(path.string(),
                                               specfem::attenuation::Setup{}));
  std::filesystem::remove(path);
}

TEST(GlobeMeshReader, RejectsOceanLoadOffFreeSurface) {
  const auto path = globe_reader_test_impl::write_database(
      { .oceans = true, .ocean_faces = { { 1, 1 } } });
  globe_reader_test_impl::expect_read_throws_with(path,
                                                  "not on the free surface");
  std::filesystem::remove(path);
}

TEST(GlobeMeshReader, RejectsOceanLoadWithoutOceans) {
  const auto path =
      globe_reader_test_impl::write_database({ .ocean_faces = { { 1, 3 } } });
  globe_reader_test_impl::expect_read_throws_with(path, "oceans are disabled");
  std::filesystem::remove(path);
}
