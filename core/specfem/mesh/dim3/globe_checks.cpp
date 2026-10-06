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

void specfem::mesh::globe_impl::check_interfaces_match_medium_contrast(
    const specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
        &adjacency_graph,
    const std::vector<specfem::mesh::globe_impl::named_surface> &surfaces) {

  // A face identified by its mesh element and orientation.
  using FaceKey = std::pair<int, specfem::mesh_entity::dim3::type>;

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

  const auto describe = [](const std::set<FaceKey> &faces,
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
  };

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
            << "):" << describe(missing, 10) << "\n";
  }
  if (!unexpected.empty()) {
    message << "  Implied but not recorded (" << unexpected.size()
            << "):" << describe(unexpected, 10) << "\n";
  }
  message << "  Element numbers are one-based, as in the database.";
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
  // Each check compares two database sections that record the same fact.
  specfem::mesh::globe_impl::check_interfaces_match_medium_contrast(
      this->adjacency_graph,
      { { "CMB", &this->globe.cmb }, { "ICB", &this->globe.icb } });
}

void specfem::mesh::mesh<specfem::simulation::model::Globe3D>::check_supported()
    const {
  specfem::mesh::globe_impl::check_no_cross_rank_fluid_solid_faces(
      this->adjacency_graph, this->globe.element_context);
}
