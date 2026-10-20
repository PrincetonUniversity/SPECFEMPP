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

void specfem::mesh::globe_impl::check_interface_surfaces_match_medium_tags(
    const specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
        &adjacency_graph,
    const std::vector<specfem::mesh::globe_element_context> &element_context,
    const std::vector<specfem::mesh::globe_impl::InterfaceSurface> &surfaces) {

  // A face identified by its mesh element and orientation.
  using FaceKey = std::pair<int, specfem::mesh_entity::dim3::type>;

  // Attribute every implied face to the surface separating its two regions.
  std::vector<std::set<FaceKey>> implied(surfaces.size());
  std::set<FaceKey> unattributed;
  const auto &graph = adjacency_graph.local_connections();
  for (const auto v : boost::make_iterator_range(boost::vertices(graph))) {
    for (const auto e :
         boost::make_iterator_range(boost::out_edges(v, graph))) {
      const auto &edge = graph[e];
      if (edge.connection !=
              specfem::element_connections::type::weakly_conforming ||
          !specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                          edge.orientation)) {
        continue;
      }
      const FaceKey key = { static_cast<int>(v), edge.orientation };
      const auto source_region = element_context.at(v).region;
      const auto target_region =
          element_context.at(boost::target(e, graph)).region;
      const auto surface = std::find_if(
          surfaces.begin(), surfaces.end(),
          [&](const specfem::mesh::globe_impl::InterfaceSurface &candidate) {
            const auto &[first, second] = candidate.regions;
            return (first == source_region && second == target_region) ||
                   (first == target_region && second == source_region);
          });
      if (surface == surfaces.end()) {
        unattributed.insert(key);
      } else {
        implied[std::distance(surfaces.begin(), surface)].insert(key);
      }
    }
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

  std::ostringstream mismatches;
  for (std::size_t isurface = 0; isurface < surfaces.size(); ++isurface) {
    const auto &[name, surface, regions] = surfaces[isurface];
    std::set<FaceKey> recorded;
    for (std::size_t iface = 0; iface < surface->elements.size(); ++iface) {
      recorded.insert({ surface->elements[iface], surface->faces[iface] });
    }

    std::set<FaceKey> missing;
    std::set<FaceKey> unexpected;
    std::set_difference(recorded.begin(), recorded.end(),
                        implied[isurface].begin(), implied[isurface].end(),
                        std::inserter(missing, missing.begin()));
    std::set_difference(implied[isurface].begin(), implied[isurface].end(),
                        recorded.begin(), recorded.end(),
                        std::inserter(unexpected, unexpected.begin()));
    if (missing.empty() && unexpected.empty()) {
      continue;
    }

    mismatches << "  " << name << " ("
               << specfem::element::to_string(regions.first) << " / "
               << specfem::element::to_string(regions.second)
               << "): " << implied[isurface].size() << " faces implied, "
               << recorded.size() << " recorded\n";
    if (!missing.empty()) {
      mismatches << "    Recorded but not implied (" << missing.size()
                 << "):" << describe(missing, 10) << "\n";
    }
    if (!unexpected.empty()) {
      mismatches << "    Implied but not recorded (" << unexpected.size()
                 << "):" << describe(unexpected, 10) << "\n";
    }
  }
  if (!unattributed.empty()) {
    mismatches << "  Implied between regions no surface separates ("
               << unattributed.size() << "):" << describe(unattributed, 10)
               << "\n";
  }

  if (mismatches.str().empty()) {
    return;
  }

  throw std::runtime_error(
      "Globe mesh database is inconsistent: the interface faces implied by "
      "medium tags and adjacency differ from the recorded interface "
      "surfaces.\n" +
      mismatches.str() +
      "  Element numbers are one-based, as in the database.");
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
  using specfem::element::region_tag;

  // Each check compares two database sections that record the same fact.
  specfem::mesh::globe_impl::check_interface_surfaces_match_medium_tags(
      this->adjacency_graph, this->globe.element_context,
      { { "CMB",
          &this->globe.cmb,
          { region_tag::crust_mantle, region_tag::outer_core } },
        { "ICB",
          &this->globe.icb,
          { region_tag::outer_core, region_tag::inner_core } } });
}

void specfem::mesh::mesh<specfem::simulation::model::Globe3D>::check_supported()
    const {
  specfem::mesh::globe_impl::check_no_cross_rank_fluid_solid_faces(
      this->adjacency_graph, this->globe.element_context);
}
