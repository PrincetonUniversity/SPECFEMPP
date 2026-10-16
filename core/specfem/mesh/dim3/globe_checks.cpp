#include "globe_checks.hpp"

#include "specfem/element.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

std::string specfem::mesh::globe_impl::describe_faces(
    const std::set<specfem::mesh::globe_impl::face_key> &faces,
    const std::size_t limit) {
  std::ostringstream text;
  std::size_t shown = 0;
  for (const auto &[ispec, face] : faces) {
    if (shown++ == limit) {
      text << " ... (" << (faces.size() - limit) << " more)";
      break;
    }
    text << " (element " << (ispec + 1)
         << ", face=" << specfem::mesh_entity::dim3::to_string(face) << ")";
  }
  return text.str();
}

void specfem::mesh::globe_impl::check_interfaces_match_medium_contrast(
    const specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
        &adjacency_graph,
    const std::vector<specfem::mesh::globe_impl::named_surface> &surfaces) {
  using FaceKey = specfem::mesh::globe_impl::face_key;

  std::set<FaceKey> implied;
  const auto &graph = adjacency_graph.local_connections();
  for (const auto v : boost::make_iterator_range(boost::vertices(graph))) {
    for (const auto e :
         boost::make_iterator_range(boost::out_edges(v, graph))) {
      const auto &edge = graph[e];
      if (edge.connection ==
              specfem::element_connections::type::weakly_conforming &&
          specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                         edge.orientation)) {
        implied.insert({ static_cast<int>(v), edge.orientation });
      }
    }
  }

  std::set<FaceKey> recorded;
  for (const auto &[name, surface] : surfaces) {
    for (std::size_t iface = 0; iface < surface->elements.size(); ++iface) {
      recorded.insert({ surface->elements[iface], surface->faces[iface] });
    }
  }

  std::set<FaceKey> missing;
  std::set<FaceKey> unexpected;
  std::set_difference(recorded.begin(), recorded.end(), implied.begin(),
                      implied.end(), std::inserter(missing, missing.begin()));
  std::set_difference(implied.begin(), implied.end(), recorded.begin(),
                      recorded.end(),
                      std::inserter(unexpected, unexpected.begin()));

  if (missing.empty() && unexpected.empty()) {
    return;
  }

  std::ostringstream message;
  message << "Globe mesh database is inconsistent: the interface faces implied "
             "by medium tags and adjacency differ from the recorded interface "
             "surfaces.\n"
          << "  Implied by medium contrast: " << implied.size() << " faces\n"
          << "  Recorded:                   " << recorded.size() << " faces (";
  for (std::size_t isurface = 0; isurface < surfaces.size(); ++isurface) {
    message << (isurface == 0 ? "" : " + ")
            << surfaces[isurface].second->elements.size() << " "
            << surfaces[isurface].first;
  }
  message << ")\n";
  if (!missing.empty()) {
    message << "  Recorded but not implied (" << missing.size()
            << "):" << specfem::mesh::globe_impl::describe_faces(missing, 10)
            << "\n";
  }
  if (!unexpected.empty()) {
    message << "  Implied but not recorded (" << unexpected.size()
            << "):" << specfem::mesh::globe_impl::describe_faces(unexpected, 10)
            << "\n";
  }
  message << "  Element numbers are one-based, as in the database.";
  throw std::runtime_error(message.str());
}

void specfem::mesh::globe_impl::check_chunk_count(const int nchunks) {
  if (nchunks != 1 && nchunks != 2 && nchunks != 3 && nchunks != 6) {
    throw std::runtime_error(
        "Globe mesh database is inconsistent: it records " +
        std::to_string(nchunks) +
        " chunks, but the mesher only produces 1, 2, 3 or 6.");
  }
}

void specfem::mesh::globe_impl::check_absorbing_matches_chunk_count(
    const int nchunks,
    const specfem::mesh::absorbing_boundary<
        specfem::element::dimension_tag::dim3> &absorbing_boundary,
    const std::vector<specfem::mesh::globe_element_context> &element_context) {
  const int nfaces = absorbing_boundary.nelements;
  if (nfaces == 0) {
    return;
  }

  if (nchunks == 6 || nchunks == 3) {
    throw std::runtime_error(
        "Globe mesh database is inconsistent: it has " +
        std::to_string(nfaces) + " absorbing (Stacey) faces with " +
        std::to_string(nchunks) + " chunks. The mesher " +
        (nchunks == 6 ? "cannot place absorbing conditions on the full Earth."
                      : "does not support absorbing conditions for 3 "
                        "chunks."));
  }

  for (int iface = 0; iface < nfaces; ++iface) {
    const int ispec = absorbing_boundary.index_mapping(iface);
    if (element_context.at(ispec).region ==
        specfem::element::region_tag::inner_core) {
      throw std::runtime_error(
          "Globe mesh database is inconsistent: absorbing (Stacey) face on "
          "inner-core element " +
          std::to_string(ispec + 1) +
          ". The mesher places Stacey faces on the crust/mantle and outer "
          "core only.");
    }
  }
}

void specfem::mesh::globe_impl::check_surface_is_subset(
    const specfem::mesh::globe_impl::named_surface &subset,
    const specfem::mesh::globe_impl::named_surface &superset) {
  using FaceKey = specfem::mesh::globe_impl::face_key;

  std::set<FaceKey> contained;
  const auto &[superset_name, superset_surface] = superset;
  for (std::size_t iface = 0; iface < superset_surface->elements.size();
       ++iface) {
    contained.insert(
        { superset_surface->elements[iface], superset_surface->faces[iface] });
  }

  std::set<FaceKey> outside;
  const auto &[subset_name, subset_surface] = subset;
  for (std::size_t iface = 0; iface < subset_surface->elements.size();
       ++iface) {
    const FaceKey face{ subset_surface->elements[iface],
                        subset_surface->faces[iface] };
    if (contained.count(face) == 0) {
      outside.insert(face);
    }
  }

  if (outside.empty()) {
    return;
  }

  std::ostringstream message;
  message << "Globe mesh database is inconsistent: " << outside.size() << " of "
          << subset_surface->elements.size() << " " << subset_name
          << " faces are not on the " << superset_name << ":"
          << specfem::mesh::globe_impl::describe_faces(outside, 10)
          << "\n  Element numbers are one-based, as in the database.";
  throw std::runtime_error(message.str());
}

void specfem::mesh::globe_impl::check_surface_faces(
    const specfem::mesh::globe_impl::named_surface &surface,
    const specfem::element::region_tag region,
    const specfem::mesh_entity::dim3::type face,
    const std::vector<specfem::mesh::globe_element_context> &element_context) {
  const auto &[name, entries] = surface;

  std::set<specfem::mesh::globe_impl::face_key> misplaced;
  for (std::size_t iface = 0; iface < entries->elements.size(); ++iface) {
    const int ispec = entries->elements[iface];
    if (entries->faces[iface] != face ||
        element_context.at(ispec).region != region) {
      misplaced.insert({ ispec, entries->faces[iface] });
    }
  }

  if (misplaced.empty()) {
    return;
  }

  std::ostringstream message;
  message << "Globe mesh database is inconsistent: " << misplaced.size()
          << " of " << entries->elements.size() << " " << name
          << " faces are not the "
          << specfem::mesh_entity::dim3::to_string(face) << " face of a "
          << specfem::element::to_string(region) << " element:"
          << specfem::mesh::globe_impl::describe_faces(misplaced, 10)
          << "\n  Element numbers are one-based, as in the database.";
  throw std::runtime_error(message.str());
}

void specfem::mesh::globe_impl::check_no_cross_rank_fluid_solid_faces(
    const specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
        &adjacency_graph,
    const std::vector<specfem::mesh::globe_element_context> &element_context) {
  using specfem::mesh_entity::dim3::type;

  for (const auto &connection : adjacency_graph.mpi_connections()) {
    // An outer-core element with a radial face on an MPI boundary means either
    // a cross-rank fluid-solid face or a radial split inside the fluid, which
    // voids the lateral-partitioning assumption this relies on. Radial MPI
    // faces on solid elements, and shared edges and corners, are expected.
    const auto face = connection.orientation;
    const bool is_radial_face = (face == type::top || face == type::bottom);
    const auto ispec = connection.local_index;
    const bool is_fluid = element_context.at(ispec).region ==
                          specfem::element::region_tag::outer_core;
    if (is_radial_face && is_fluid) {
      throw std::runtime_error(
          "Unsupported globe mesh: outer-core element " +
          std::to_string(ispec + 1) + " has a radial (" +
          specfem::mesh_entity::dim3::to_string(face) +
          ") face on an MPI partition boundary. The CMB and ICB must lie "
          "within a rank: a fluid-solid interface split across ranks would be "
          "dropped from the coupling silently. Supporting such a mesh requires "
          "promoting MPI connections in setup_coupled_interfaces and "
          "collecting cross-rank face pairs in element_intersections.");
    }
  }
}

void specfem::mesh::mesh<
    specfem::simulation::model::Globe3D>::check_consistency() const {
  // Each check holds the database to a rule the mesher guarantees: either two
  // sections record the same fact, or a section respects a meshing restriction.
  const auto &model_config = this->globe.model_config;
  specfem::mesh::globe_impl::check_chunk_count(model_config.nchunks);

  specfem::mesh::globe_impl::check_interfaces_match_medium_contrast(
      this->adjacency_graph,
      { { "CMB", &this->globe.cmb }, { "ICB", &this->globe.icb } });

  specfem::mesh::globe_impl::check_absorbing_matches_chunk_count(
      model_config.nchunks, this->boundaries.absorbing_boundary,
      this->globe.element_context);

  // The mesher writes the top faces of the crust/mantle as the free surface.
  specfem::mesh::globe_impl::check_surface_faces(
      { "free surface", &this->globe.free_surface },
      specfem::element::region_tag::crust_mantle,
      specfem::mesh_entity::dim3::type::top, this->globe.element_context);

  // The mesher writes the ocean load only when oceans are enabled, and then
  // on the free surface of the crust/mantle.
  if (!model_config.oceans && !this->globe.ocean_load.elements.empty()) {
    throw std::runtime_error(
        "Globe mesh database is inconsistent: oceans are disabled but " +
        std::to_string(this->globe.ocean_load.elements.size()) +
        " ocean-load faces are recorded.");
  }
  specfem::mesh::globe_impl::check_surface_is_subset(
      { "ocean-load", &this->globe.ocean_load },
      { "free surface", &this->globe.free_surface });
}

void specfem::mesh::mesh<specfem::simulation::model::Globe3D>::check_supported()
    const {
  specfem::mesh::globe_impl::check_no_cross_rank_fluid_solid_faces(
      this->adjacency_graph, this->globe.element_context);
}
