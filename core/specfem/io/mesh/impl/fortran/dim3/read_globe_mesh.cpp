#include "specfem/io/mesh/impl/fortran/dim3/read_globe_mesh.hpp"

#include "specfem/attenuation.hpp"
#include "specfem/io.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/io/mesh/impl/fortran/dim3/globe_hex27.hpp"
#include "specfem/medium_container.hpp"
#include "specfem/mpi.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <boost/graph/adjacency_list.hpp>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace specfem::io::mesh::impl::fortran::dim3_impl {

constexpr int globe_database_version_min = 2;
constexpr int globe_database_version_max = 2;
constexpr int material_oracle = 1;
constexpr int medium_acoustic = 1;
constexpr int medium_elastic = 2;

void check_stream(const std::ifstream &stream, const std::string &section) {
  if (!stream) {
    throw std::runtime_error("Failed to read globe mesh database section: " +
                             section);
  }
}

std::pair<std::string, int> read_magic(std::ifstream &stream) {
  int record_size = 0;
  int trailing_size = 0;
  int version = 0;
  std::array<char, 32> magic{};

  stream.read(reinterpret_cast<char *>(&record_size), sizeof(record_size));
  if (record_size != static_cast<int>(magic.size() + sizeof(version))) {
    throw std::runtime_error("Invalid SPECFEM++ globe database header size");
  }
  stream.read(magic.data(), magic.size());
  stream.read(reinterpret_cast<char *>(&version), sizeof(version));
  stream.read(reinterpret_cast<char *>(&trailing_size), sizeof(trailing_size));
  check_stream(stream, "header");
  if (trailing_size != record_size) {
    throw std::runtime_error("Mismatched Fortran record markers in globe "
                             "database header");
  }

  std::string result(magic.data(), magic.size());
  const auto last = result.find_last_not_of(' ');
  result.resize(last == std::string::npos ? 0 : last + 1);
  return { result, version };
}

std::vector<int> read_counted_ints(std::ifstream &stream,
                                   const std::string &section) {
  int record_size = 0;
  int trailing_size = 0;
  int count = 0;
  stream.read(reinterpret_cast<char *>(&record_size), sizeof(record_size));
  stream.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (count < 0 || record_size != static_cast<int>((count + 1) * sizeof(int))) {
    throw std::runtime_error("Invalid " + section +
                             " record in globe mesh database");
  }
  std::vector<int> values(count);
  if (count > 0) {
    stream.read(reinterpret_cast<char *>(values.data()), count * sizeof(int));
  }
  stream.read(reinterpret_cast<char *>(&trailing_size), sizeof(trailing_size));
  check_stream(stream, section);
  if (trailing_size != record_size) {
    throw std::runtime_error("Mismatched Fortran record markers in " + section);
  }
  return values;
}

std::vector<bool> read_counted_logicals(std::ifstream &stream,
                                        const std::string &section) {
  const auto raw = read_counted_ints(stream, section);
  std::vector<bool> values(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    values[i] = raw[i] != 0;
  }
  return values;
}

std::string read_fixed_string(std::ifstream &stream,
                              const std::string &section) {
  int record_size = 0;
  int trailing_size = 0;
  stream.read(reinterpret_cast<char *>(&record_size), sizeof(record_size));
  if (record_size < 0) {
    throw std::runtime_error("Invalid " + section +
                             " record in globe mesh database");
  }
  std::string value(static_cast<std::size_t>(record_size), ' ');
  if (record_size > 0) {
    stream.read(value.data(), record_size);
  }
  stream.read(reinterpret_cast<char *>(&trailing_size), sizeof(trailing_size));
  check_stream(stream, section);
  if (trailing_size != record_size) {
    throw std::runtime_error("Mismatched Fortran record markers in " + section);
  }
  const auto last = value.find_last_not_of(" \0", std::string::npos, 2);
  value.resize(last == std::string::npos ? 0 : last + 1);
  return value;
}

specfem::mesh::globe_boundary_surface read_surface(std::ifstream &stream,
                                                   const int nspec) {
  specfem::mesh::globe_boundary_surface result;
  int nfaces = 0;
  specfem::io::fortran_read_line(stream, &nfaces);
  if (nfaces < 0) {
    throw std::runtime_error("Negative face count in globe mesh database");
  }
  result.elements.resize(nfaces);
  std::vector<int> faces(nfaces);
  if (nfaces > 0) {
    specfem::io::fortran_read_line(stream, &result.elements, &faces);
  }
  result.faces.resize(nfaces);
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      element_view(result.elements.data(), nfaces),
      faces_view(faces.data(), nfaces);
  Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::LayoutLeft,
               Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      surface_face_view(result.faces.data(), nfaces);
  int invalid_boundary_count = 0;
  Kokkos::parallel_reduce(
      "specfem::io::mesh::dim3::read_globe_mesh::read_surface",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nfaces),
      [=](const int iface, int &local_invalid_boundary_count) {
        if (element_view(iface) < 1 || element_view(iface) > nspec ||
            faces_view(iface) < 1 || faces_view(iface) > 6) {
          ++local_invalid_boundary_count;
          return;
        }
        --element_view(iface);
        surface_face_view(iface) =
            static_cast<specfem::mesh_entity::dim3::type>(faces_view(iface));
      },
      invalid_boundary_count);
  if (invalid_boundary_count > 0) {
    throw std::runtime_error("Invalid boundary entry in globe mesh database");
  }
  return result;
}

void set_coordinate_bounds(
    specfem::mesh::control_nodes<specfem::element::dimension_tag::dim3>
        &nodes) {
  nodes.xmin = nodes.ymin = nodes.zmin = std::numeric_limits<type_real>::max();
  nodes.xmax = nodes.ymax = nodes.zmax =
      std::numeric_limits<type_real>::lowest();
  for (int inode = 0; inode < nodes.nnodes; ++inode) {
    nodes.xmin = std::min(nodes.xmin, nodes.coordinates(inode, 0));
    nodes.xmax = std::max(nodes.xmax, nodes.coordinates(inode, 0));
    nodes.ymin = std::min(nodes.ymin, nodes.coordinates(inode, 1));
    nodes.ymax = std::max(nodes.ymax, nodes.coordinates(inode, 1));
    nodes.zmin = std::min(nodes.zmin, nodes.coordinates(inode, 2));
    nodes.zmax = std::max(nodes.zmax, nodes.coordinates(inode, 2));
  }
}

int corner_entity(const int local_corner) {
  constexpr std::array<int, 8> entities = { 19, 20, 22, 21, 23, 24, 26, 25 };
  return entities.at(local_corner);
}

int entity_from_corners(const std::set<int> &corners) {
  static const std::array<std::pair<int, std::set<int>>, 26> entities = {
    { { 1, { 0, 1, 2, 3 } }, { 2, { 1, 2, 5, 6 } }, { 3, { 4, 5, 6, 7 } },
      { 4, { 0, 3, 4, 7 } }, { 5, { 0, 1, 4, 5 } }, { 6, { 2, 3, 6, 7 } },
      { 7, { 0, 3 } },       { 8, { 1, 2 } },       { 9, { 5, 6 } },
      { 10, { 4, 7 } },      { 11, { 0, 1 } },      { 12, { 4, 5 } },
      { 13, { 0, 4 } },      { 14, { 1, 5 } },      { 15, { 2, 3 } },
      { 16, { 6, 7 } },      { 17, { 3, 7 } },      { 18, { 2, 6 } },
      { 19, { 0 } },         { 20, { 1 } },         { 21, { 3 } },
      { 22, { 2 } },         { 23, { 4 } },         { 24, { 5 } },
      { 25, { 7 } },         { 26, { 6 } } }
  };
  for (const auto &[entity, entity_corners] : entities) {
    if (corners == entity_corners) {
      return entity;
    }
  }
  throw std::runtime_error(
      "MPI anchor nodes do not form a hex face, edge, or corner");
}

struct mpi_element_description {
  int local_index = -1;
  int orientation = 0;
  int nshared = 0;
  int anchor = 0;
  std::array<double, 12> coordinates{};
};

std::vector<mpi_element_description> describe_mpi_interface(
    const specfem::mesh::globe_mpi_interface &interface,
    const specfem::mesh::control_nodes<specfem::element::dimension_tag::dim3>
        &nodes) {
  std::set<int> shared_nodes(interface.node_ids.begin(),
                             interface.node_ids.end());
  std::vector<mpi_element_description> descriptions;
  for (int ispec = 0; ispec < nodes.nspec; ++ispec) {
    std::set<int> local_corners;
    struct Point {
      std::array<double, 3> xyz;
      int local_corner;
      bool operator<(const Point &other) const { return xyz < other.xyz; }
    };
    std::vector<Point> points;
    for (int corner = 0; corner < 8; ++corner) {
      const int inode = nodes.control_node_index(ispec, corner);
      if (shared_nodes.contains(inode)) {
        local_corners.insert(corner);
        points.push_back(
            { { nodes.coordinates(inode, 0), nodes.coordinates(inode, 1),
                nodes.coordinates(inode, 2) },
              corner });
      }
    }
    if (points.empty()) {
      continue;
    }
    if (points.size() != 1 && points.size() != 2 && points.size() != 4) {
      throw std::runtime_error(
          "Invalid number of shared MPI corners on globe element " +
          std::to_string(ispec));
    }
    std::sort(points.begin(), points.end());
    mpi_element_description description;
    description.local_index = ispec;
    description.orientation = entity_from_corners(local_corners);
    description.nshared = static_cast<int>(points.size());
    description.anchor = corner_entity(points.front().local_corner);
    for (int i = 0; i < description.nshared; ++i) {
      for (int component = 0; component < 3; ++component) {
        description.coordinates[3 * i + component] = points[i].xyz[component];
      }
    }
    descriptions.push_back(description);
  }
  return descriptions;
}

bool same_interface_entity(const mpi_element_description &left,
                           const mpi_element_description &right) {
  if (left.nshared != right.nshared) {
    return false;
  }
  constexpr double coordinate_tolerance = 1.0e-4;
  for (int i = 0; i < 3 * left.nshared; ++i) {
    if (std::abs(left.coordinates[i] - right.coordinates[i]) >
        coordinate_tolerance) {
      return false;
    }
  }
  return true;
}

void build_mpi_adjacency(specfem::mesh::globe3d_mesh &mesh) {
  if (mesh.globe.mpi_interfaces.empty()) {
    return;
  }
#ifndef SPECFEM_ENABLE_MPI
  throw std::runtime_error(
      "A partitioned globe database requires an MPI-enabled build");
#else
  auto &interfaces = mesh.globe.mpi_interfaces;
  const int ninterfaces = static_cast<int>(interfaces.size());
  std::vector<std::vector<mpi_element_description>> local(ninterfaces);
  std::vector<std::vector<mpi_element_description>> remote(ninterfaces);
  std::vector<int> send_counts(ninterfaces), receive_counts(ninterfaces);
  std::vector<MPI_Request> requests(2 * ninterfaces, MPI_REQUEST_NULL);
  const auto comm = specfem::MPI::communicator();

  for (int i = 0; i < ninterfaces; ++i) {
    local[i] = describe_mpi_interface(interfaces[i], mesh.control_nodes);
    send_counts[i] = static_cast<int>(local[i].size());
    SPECFEM_MPI_SAFECALL(MPI_Irecv(&receive_counts[i], 1, MPI_INT,
                                   interfaces[i].neighbor_rank, 29001, comm,
                                   &requests[2 * i]));
    SPECFEM_MPI_SAFECALL(MPI_Isend(&send_counts[i], 1, MPI_INT,
                                   interfaces[i].neighbor_rank, 29001, comm,
                                   &requests[2 * i + 1]));
  }
  SPECFEM_MPI_SAFECALL(MPI_Waitall(static_cast<int>(requests.size()),
                                   requests.data(), MPI_STATUSES_IGNORE));

  std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);
  for (int i = 0; i < ninterfaces; ++i) {
    remote[i].resize(receive_counts[i]);
    const std::size_t send_bytes =
        local[i].size() * sizeof(mpi_element_description);
    const std::size_t receive_bytes =
        remote[i].size() * sizeof(mpi_element_description);
    if (send_bytes > std::numeric_limits<int>::max() ||
        receive_bytes > std::numeric_limits<int>::max()) {
      throw std::runtime_error("Globe MPI interface description is too large");
    }
    SPECFEM_MPI_SAFECALL(
        MPI_Irecv(remote[i].data(), static_cast<int>(receive_bytes), MPI_BYTE,
                  interfaces[i].neighbor_rank, 29002, comm, &requests[2 * i]));
    SPECFEM_MPI_SAFECALL(MPI_Isend(
        local[i].data(), static_cast<int>(send_bytes), MPI_BYTE,
        interfaces[i].neighbor_rank, 29002, comm, &requests[2 * i + 1]));
  }
  SPECFEM_MPI_SAFECALL(MPI_Waitall(static_cast<int>(requests.size()),
                                   requests.data(), MPI_STATUSES_IGNORE));

  for (int i = 0; i < ninterfaces; ++i) {
    std::vector<bool> matched(remote[i].size(), false);
    for (const auto &local_element : local[i]) {
      int match = -1;
      for (int candidate = 0; candidate < static_cast<int>(remote[i].size());
           ++candidate) {
        if (!matched[candidate] &&
            same_interface_entity(local_element, remote[i][candidate])) {
          if (match != -1) {
            throw std::runtime_error(
                "Ambiguous element match across globe MPI interface");
          }
          match = candidate;
        }
      }
      if (match == -1) {
        throw std::runtime_error(
            "Could not match an element across globe MPI interface");
      }
      matched[match] = true;
      const auto &remote_element = remote[i][match];
      mesh.adjacency_graph.mpi_connections().emplace_back(
          specfem::element_connections::type::strongly_conforming,
          static_cast<specfem::mesh_entity::dim3::type>(
              local_element.orientation),
          interfaces[i].neighbor_rank,
          static_cast<specfem::mesh_entity::dim3::type>(
              remote_element.orientation),
          local_element.local_index, remote_element.local_index,
          static_cast<specfem::mesh_entity::dim3::type>(local_element.anchor),
          static_cast<specfem::mesh_entity::dim3::type>(remote_element.anchor));
    }
    if (std::find(matched.begin(), matched.end(), false) != matched.end()) {
      throw std::runtime_error(
          "Neighbor reported unmatched elements on globe MPI interface");
    }
  }
#endif
}

specfem::mesh::materials<specfem::element::dimension_tag::dim3>
make_materials(const std::vector<int> &medium_tags,
               const std::vector<int> &property_tags,
               const bool attenuation_enabled) {
  using Dimension = specfem::element::dimension_tag;
  using Medium = specfem::element::medium_tag;
  using Property = specfem::element::property_tag;
  using Attenuation = specfem::element::attenuation_tag;
  using Materials = specfem::mesh::materials<Dimension::dim3>;

  Materials materials;
  materials.nspec = static_cast<int>(medium_tags.size());
  materials.material_index_mapping.resize(materials.nspec);

  specfem::medium_container::material<Dimension::dim3, Medium::acoustic,
                                      Property::isotropic, Attenuation::none>
      acoustic(1.0, 1.0, 0.0);
  const int acoustic_index = materials.add_material(acoustic);

  specfem::medium_container::material<Dimension::dim3, Medium::elastic,
                                      Property::isotropic, Attenuation::none>
      elastic(1.0, 1.0, 2.0, 0.0);
  const int elastic_index = materials.add_material(elastic);

  std::optional<int> attenuating_elastic_index;
  if (attenuation_enabled) {
    specfem::medium_container::material<Dimension::dim3, Medium::elastic,
                                        Property::isotropic,
                                        Attenuation::constant_isotropic>
        attenuating_elastic(1.0, 1.0, 2.0, 9999.0, 9999.0, 0.0);
    attenuating_elastic_index = materials.add_material(attenuating_elastic);
  }

  for (int ispec = 0; ispec < materials.nspec; ++ispec) {
    if (property_tags[ispec] != 0) {
      throw std::runtime_error(
          "The globe database contains anisotropic/TISO elements, but "
          "SPECFEM++ has no 3-D anisotropic property container or kernel yet");
    }
    if (medium_tags[ispec] == medium_acoustic) {
      materials.material_index_mapping[ispec] = { Medium::acoustic,
                                                  Property::isotropic,
                                                  Attenuation::none,
                                                  acoustic_index, ispec };
    } else if (medium_tags[ispec] == medium_elastic) {
      const auto attenuation = attenuation_enabled
                                   ? Attenuation::constant_isotropic
                                   : Attenuation::none;
      const int index =
          attenuation_enabled ? *attenuating_elastic_index : elastic_index;
      materials.material_index_mapping[ispec] = { Medium::elastic,
                                                  Property::isotropic,
                                                  attenuation, index, ispec };
    } else {
      throw std::runtime_error("Unknown medium tag in globe mesh database");
    }
  }
  return materials;
}

} // namespace specfem::io::mesh::impl::fortran::dim3_impl

specfem::mesh::globe3d_mesh
specfem::io::mesh::impl::fortran::dim3::read_globe_mesh(
    const std::string &database_file,
    const specfem::attenuation::Setup &attenuation_setup) {
  namespace reader_impl = specfem::io::mesh::impl::fortran::dim3_impl;
  using Dimension = specfem::element::dimension_tag;

  std::ifstream stream(database_file, std::ios::in | std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("Could not open globe mesh database: " +
                             database_file);
  }

  const auto [magic, version] = reader_impl::read_magic(stream);
  if (magic != "SPECFEMPP_GLOBE_DB") {
    throw std::runtime_error("Not a SPECFEM++ globe mesh database: " +
                             database_file);
  }
  if (version < reader_impl::globe_database_version_min ||
      version > reader_impl::globe_database_version_max) {
    throw std::runtime_error("Unsupported globe mesh database version " +
                             std::to_string(version));
  }

  specfem::mesh::globe3d_mesh mesh;
  auto &globe = mesh.globe;
  globe.format_version = version;

  specfem::io::fortran_read_line(stream, &globe.model_config.planet_type,
                                 &globe.planet_radius, &globe.average_density);

  int ngnod = 0;
  specfem::io::fortran_read_line(stream, &ngnod, &mesh.element_grid.ngllx,
                                 &mesh.element_grid.nglly,
                                 &mesh.element_grid.ngllz, &globe.nregions);
  if (ngnod != 27) {
    throw std::runtime_error("Globe mesh database must contain hex27 anchors");
  }

  auto &model_config = globe.model_config;
  specfem::io::fortran_read_line(
      stream, &model_config.ellipticity, &model_config.topography,
      &model_config.gravity, &globe.full_gravity, &model_config.rotation,
      &model_config.attenuation, &model_config.oceans,
      &globe.has_reference_geometry);
  specfem::io::fortran_read_line(stream, &globe.material_mode);
  if (globe.material_mode != reader_impl::material_oracle) {
    throw std::runtime_error(
        "Only oracle-backed globe databases are supported");
  }

  model_config.model_name =
      reader_impl::read_fixed_string(stream, "model name");
  globe.model_verification.codes =
      reader_impl::read_counted_ints(stream, "model codes");
  globe.model_verification.flags =
      reader_impl::read_counted_logicals(stream, "model flags");
  specfem::io::fortran_read_line(stream, &model_config.nchunks,
                                 &model_config.nex_xi, &model_config.nex_eta);
  specfem::io::fortran_read_line(
      stream, &model_config.min_attenuation_period,
      &model_config.max_attenuation_period,
      &globe.model_verification.attenuation_source_frequency);
  model_config.validate();
  if (model_config.attenuation) {
    const double expected_source_frequency =
        1.0 / std::sqrt(model_config.min_attenuation_period *
                        model_config.max_attenuation_period);
    const double source_frequency_error =
        std::abs(globe.model_verification.attenuation_source_frequency -
                 expected_source_frequency);
    if (source_frequency_error >
        1.0e-12 * std::abs(expected_source_frequency)) {
      throw std::runtime_error(
          "Globe mesh database attenuation period band failed its central "
          "frequency check");
    }
  }

  int nnode = 0;
  specfem::io::fortran_read_line(stream, &nnode);
  if (nnode <= 0) {
    throw std::runtime_error("Globe mesh database contains no anchor nodes");
  }
  mesh.control_nodes = { ngnod, nnode };
  std::vector<double> x(nnode), y(nnode), z(nnode);
  specfem::io::fortran_read_line(stream, &x, &y, &z);
  auto coordinates = mesh.control_nodes.coordinates;
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      x_view(x.data(), nnode), y_view(y.data(), nnode), z_view(z.data(), nnode);
  Kokkos::parallel_for(
      "specfem::io::mesh::dim3::read_globe_mesh::copy_coordinates",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nnode),
      [=](const int inode) {
        coordinates(inode, 0) = x_view(inode);
        coordinates(inode, 1) = y_view(inode);
        coordinates(inode, 2) = z_view(inode);
      });
  Kokkos::fence();
  reader_impl::set_coordinate_bounds(mesh.control_nodes);

  globe.reference_coordinates =
      specfem::mesh::globe_mesh_data::CoordinatesViewType(
          "specfem::mesh::globe_reference_coordinates", nnode);
  if (globe.has_reference_geometry) {
    specfem::io::fortran_read_line(stream, &x, &y, &z);
  }
  const bool has_reference_geometry = globe.has_reference_geometry;
  auto reference_coordinates = globe.reference_coordinates;
  Kokkos::parallel_for(
      "specfem::io::mesh::dim3::read_globe_mesh::copy_reference_coordinates",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nnode),
      [=](const int inode) {
        reference_coordinates(inode, 0) =
            has_reference_geometry ? x_view(inode) : coordinates(inode, 0);
        reference_coordinates(inode, 1) =
            has_reference_geometry ? y_view(inode) : coordinates(inode, 1);
        reference_coordinates(inode, 2) =
            has_reference_geometry ? z_view(inode) : coordinates(inode, 2);
      });
  Kokkos::fence();

  specfem::io::fortran_read_line(stream, &mesh.nspec);
  if (mesh.nspec <= 0) {
    throw std::runtime_error("Globe mesh database contains no elements");
  }
  mesh.control_nodes.nspec = mesh.nspec;
  std::vector<int> regions(mesh.nspec), medium_tags(mesh.nspec),
      property_tags(mesh.nspec), idoubling(mesh.nspec);
  specfem::io::fortran_read_line(stream, &regions, &medium_tags, &property_tags,
                                 &idoubling);
  std::vector<double> rmin(mesh.nspec), rmax(mesh.nspec);
  specfem::io::fortran_read_line(stream, &rmin, &rmax);
  std::vector<bool> in_crust(mesh.nspec), in_mantle(mesh.nspec);
  specfem::io::fortran_read_line(stream, &in_crust, &in_mantle);
  std::vector<int> in_crust_values(mesh.nspec), in_mantle_values(mesh.nspec);
  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    in_crust_values[ispec] = in_crust[ispec] ? 1 : 0;
    in_mantle_values[ispec] = in_mantle[ispec] ? 1 : 0;
  }
  globe.element_context.resize(mesh.nspec);
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      regions_view(regions.data(), mesh.nspec),
      idoubling_view(idoubling.data(), mesh.nspec),
      in_crust_view(in_crust_values.data(), mesh.nspec),
      in_mantle_view(in_mantle_values.data(), mesh.nspec);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      rmin_view(rmin.data(), mesh.nspec), rmax_view(rmax.data(), mesh.nspec);
  Kokkos::View<specfem::mesh::globe_element_context *, Kokkos::LayoutLeft,
               Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      element_context_view(globe.element_context.data(), mesh.nspec);
  Kokkos::parallel_for(
      "specfem::io::mesh::dim3::read_globe_mesh::element_context",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, mesh.nspec),
      [=](const int ispec) {
        element_context_view(
            ispec) = { regions_view(ispec),       idoubling_view(ispec),
                       rmin_view(ispec),          rmax_view(ispec),
                       in_crust_view(ispec) != 0, in_mantle_view(ispec) != 0 };
      });
  Kokkos::fence();

  std::vector<int> node_ids(static_cast<std::size_t>(ngnod) * mesh.nspec);
  specfem::io::fortran_read_line(stream, &node_ids);
  mesh.control_nodes.control_node_index =
      Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace>(
          "specfem::mesh::globe_control_node_index", mesh.nspec, ngnod);
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      node_ids_view(node_ids.data(), node_ids.size());
  auto control_node_index = mesh.control_nodes.control_node_index;
  int invalid_anchor_count = 0;
  Kokkos::parallel_reduce(
      "specfem::io::mesh::dim3::read_globe_mesh::control_node_index",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
          { 0, 0 }, { mesh.nspec, ngnod }),
      [=](const int ispec, const int globe_anchor,
          int &local_invalid_anchor_count) {
        const int inode = node_ids_view(ispec * ngnod + globe_anchor) - 1;
        if (inode < 0 || inode >= nnode) {
          ++local_invalid_anchor_count;
          return;
        }
        const int specfem_anchor = globe_to_specfem_hex27[globe_anchor];
        control_node_index(ispec, specfem_anchor) = inode;
      },
      invalid_anchor_count);
  if (invalid_anchor_count > 0) {
    throw std::runtime_error("Invalid anchor node ID in globe database");
  }

  globe.free_surface = reader_impl::read_surface(stream, mesh.nspec);
  globe.cmb = reader_impl::read_surface(stream, mesh.nspec);
  globe.icb = reader_impl::read_surface(stream, mesh.nspec);
  globe.ocean_load = reader_impl::read_surface(stream, mesh.nspec);

  specfem::mesh::absorbing_boundary<Dimension::dim3> absorbing(0);
  specfem::mesh::acoustic_free_surface<Dimension::dim3> free_surface(
      static_cast<int>(globe.free_surface.elements.size()));
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      free_surface_elements_view(globe.free_surface.elements.data(),
                                 globe.free_surface.elements.size());
  Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::LayoutLeft,
               Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      free_surface_faces_view(globe.free_surface.faces.data(),
                              globe.free_surface.faces.size());
  auto free_surface_index_mapping = free_surface.index_mapping;
  auto free_surface_type = free_surface.type;
  Kokkos::parallel_for(
      "specfem::io::mesh::dim3::read_globe_mesh::free_surface",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
          0, static_cast<int>(globe.free_surface.elements.size())),
      [=](const int iface) {
        free_surface_index_mapping(iface) = free_surface_elements_view(iface);
        free_surface_type(iface) = free_surface_faces_view(iface);
      });
  Kokkos::fence();
  mesh.boundaries = { absorbing, free_surface };

  int nadjacencies = 0;
  specfem::io::fortran_read_line(stream, &nadjacencies);
  std::vector<int> xadj(mesh.nspec + 1), adjncy(nadjacencies),
      adjacency_types(nadjacencies);
  specfem::io::fortran_read_line(stream, &xadj);
  specfem::io::fortran_read_line(stream, &adjncy);
  specfem::io::fortran_read_line(stream, &adjacency_types);
  mesh.adjacency_graph =
      specfem::mesh::adjacency_graph<Dimension::dim3>(mesh.nspec);
  auto &graph = mesh.adjacency_graph.local_connections();
  using EdgeProperties =
      specfem::mesh::adjacency_graph<Dimension::dim3>::EdgeProperties;
  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    if (xadj[ispec] < 1 || xadj[ispec + 1] < xadj[ispec] ||
        xadj[ispec + 1] > nadjacencies + 1) {
      throw std::runtime_error("Invalid CSR adjacency in globe database");
    }
    for (int offset = xadj[ispec] - 1; offset < xadj[ispec + 1] - 1; ++offset) {
      const int neighbor = adjncy[offset] - 1;
      if (neighbor < 0 || neighbor >= mesh.nspec) {
        throw std::runtime_error("Invalid adjacency element in globe database");
      }
      boost::add_edge(
          ispec, neighbor,
          EdgeProperties(
              specfem::element_connections::type::strongly_conforming,
              static_cast<specfem::mesh_entity::dim3::type>(
                  adjacency_types[offset])),
          graph);
    }
  }
  mesh.adjacency_graph.assert_symmetry();

  int nneighbors = 0;
  specfem::io::fortran_read_line(stream, &nneighbors);
  globe.mpi_interfaces.resize(nneighbors);
  for (auto &interface : globe.mpi_interfaces) {
    int nshared = 0;
    specfem::io::fortran_read_line(stream, &interface.neighbor_rank, &nshared);
    interface.node_ids.resize(nshared);
    if (nshared > 0) {
      specfem::io::fortran_read_line(stream, &interface.node_ids);
      for (auto &inode : interface.node_ids) {
        --inode;
        if (inode < 0 || inode >= nnode) {
          throw std::runtime_error(
              "Invalid MPI anchor node ID in globe database");
        }
      }
    }
  }
  reader_impl::build_mpi_adjacency(mesh);
  reader_impl::check_stream(stream, "end of file");

  const bool attenuation_enabled =
      model_config.attenuation && attenuation_setup.enabled;
  mesh.materials = reader_impl::make_materials(medium_tags, property_tags,
                                               attenuation_enabled);
  mesh.tags = specfem::mesh::tags<Dimension::dim3>(
      mesh.nspec, mesh.materials, mesh.adjacency_graph, mesh.boundaries);

  mesh.setup_coupled_interfaces();
  if (attenuation_enabled) {
    if (!attenuation_setup.f0.has_value()) {
      throw std::runtime_error(
          "Globe attenuation requires an explicit reference frequency");
    }
    mesh.attenuation = {
      true, attenuation_setup.f0.value(), attenuation_setup.band,
      specfem::attenuation::compute_tau_sigma<specfem::constants::N_SLS>(
          attenuation_setup.band)
    };
    mesh.materials.apply_attenuation(mesh.attenuation);
  }

  return mesh;
}

specfem::mesh::globe3d_mesh specfem::io::read_globe_mesh(
    const std::string &database_file,
    const specfem::attenuation::Setup &attenuation_setup) {
  return specfem::io::mesh::impl::fortran::dim3::read_globe_mesh(
      database_file, attenuation_setup);
}
