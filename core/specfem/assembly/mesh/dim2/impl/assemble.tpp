#pragma once

#include "enumerations/connections.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/mesh.hpp"
#include <boost/graph/filtered_graph.hpp>

void specfem::assembly::mesh<specfem::dimension::type::dim2>::assemble(
    const specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>
        &graph) {

  const int nspec = this->nspec;
  const int ngllx = this->ngllx;
  const int ngllz = this->ngllz;

  constexpr int chunk_size = specfem::parallel_config::storage_chunk_size;

  int iglob = 0;

  // Assign unique global numbers to interior points
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    // Iterate over all interior points
    for (int iz = 1; iz < ngllz - 1; iz++) {
      for (int ix = 1; ix < ngllx - 1; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          const int ispec_mesh =
              static_cast<specfem::assembly::mesh_impl::mesh_to_compute_mapping<
                  specfem::dimension::type::dim2> &>(*this)
                  .compute_to_mesh(ispec);
          // Interior points are unique, assign unique global number
          h_index_mapping(ispec, iz, ix) = iglob;
          iglob++;
        }
      }
    }
  }

  const auto &g = graph.graph();
  // Filter out strongly conforming connections
  auto filter = [&g](const auto &edge) {
    return g[edge].connection ==
           specfem::connections::type::strongly_conforming;
  };

  // Create a filtered graph view
  const auto fg = boost::make_filtered_graph(g, filter);

  const auto element_connections =
      specfem::connections::connection_mapping(ngllx, ngllz);

  // Now lets iterate over all edges
  // We only take interior edge points, corners will be treated later
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (auto iedge : specfem::mesh_entity::edges) {
      const auto npoints =
          element_connections.number_of_points_on_orientation(iedge);
      for (int ipoint = 1; ipoint < npoints - 1; ipoint++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          const int ispec_mesh =
              static_cast<specfem::assembly::mesh_impl::mesh_to_compute_mapping<
                  specfem::dimension::type::dim2> &>(*this)
                  .compute_to_mesh(ispec);

          // Check if this point has already been assigned
          // ----
          bool previously_assigned = false;
          const auto [iz, ix] =
              element_connections.coordinates_at_edge(iedge, ipoint);

          // get all connections for this element
          for (auto edge :
               boost::make_iterator_range(boost::out_edges(ispec_mesh, fg))) {
            // Only consider the outgoing edges that match what we are looking
            // for
            if (fg[edge].orientation == iedge) {
              const int jspec_mesh = boost::target(edge, fg);
              const int jspec =
                  static_cast<
                      specfem::assembly::mesh_impl::mesh_to_compute_mapping<
                          specfem::dimension::type::dim2> &>(*this)
                      .mesh_to_compute(jspec_mesh);
              // Return edge
              const auto other_edge =
                  boost::edge(jspec_mesh, ispec_mesh, fg).first;
              const auto mapped_iedge = fg[other_edge].orientation;

              const auto [from_coords, to_coords] =
                  element_connections.map_coordinates(iedge, mapped_iedge,
                                                      ipoint);
              const auto [mapped_ix, mapped_iz] = to_coords;

              if (h_index_mapping(jspec, mapped_iz, mapped_ix) != -1) {
                h_index_mapping(ispec, iz, ix) =
                    h_index_mapping(jspec, mapped_iz, mapped_ix);
                previously_assigned = true;
                break;
              }
            }
          }
          // ----

          if (!previously_assigned) {
            h_index_mapping(ispec, iz, ix) = iglob;
            iglob++;
          }
        }
      }
    }
  }

  // Finally we need to treat corner points
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (auto corner : specfem::mesh_entity::corners) {
      for (int ielement = 0; ielement < chunk_size; ielement++) {
        int ispec = ichunk + ielement;
        if (ispec >= nspec)
          break;
        const int ispec_mesh =
            static_cast<specfem::assembly::mesh_impl::mesh_to_compute_mapping<
                specfem::dimension::type::dim2> &>(*this)
                .compute_to_mesh(ispec);

        const auto [iz, ix] = element_connections.coordinates_at_corner(corner);

        bool previously_assigned = false;

        auto valid_connections =
            specfem::mesh_entity::edges_of_corner(corner);

        valid_connections.push_back(corner);

        // get all connections for this element
        for (auto edge :
             boost::make_iterator_range(boost::out_edges(ispec_mesh, fg))) {
          // Only consider edges that contain the corner
          if (specfem::mesh_entity::contains(valid_connections,
                                             fg[edge].orientation)) {
            const int jspec_mesh = boost::target(edge, fg);
            const int jspec =
                static_cast<
                    specfem::assembly::mesh_impl::mesh_to_compute_mapping<
                        specfem::dimension::type::dim2> &>(*this)
                    .mesh_to_compute(jspec_mesh);

            // Return edge
            const auto other_edge =
                boost::edge(jspec_mesh, ispec_mesh, fg).first;
            const auto mapped_iedge = fg[other_edge].orientation;
            bool previously_assigned = false;

            // Check if the connection is an edge connection
            if (specfem::mesh_entity::contains(specfem::mesh_entity::edges,
                                               fg[edge].orientation)) {
              const auto point = element_connections.find_corner_on_edge(
                  corner, fg[edge].orientation);
              const auto [from_coords, to_coords] =
                  element_connections.map_coordinates(fg[edge].orientation,
                                                      mapped_iedge, point);
              const auto [mapped_ix, mapped_iz] = to_coords;

              if (h_index_mapping(jspec, mapped_iz, mapped_ix) != -1) {
                h_index_mapping(ispec, iz, ix) =
                    h_index_mapping(jspec, mapped_iz, mapped_ix);
                previously_assigned = true;
                break;
              }
            } else {
              // Corner to corner connection
              const auto [from_coords, to_coords] =
                  element_connections.map_coordinates(corner, mapped_iedge, 0);
              const auto [mapped_ix, mapped_iz] = to_coords;

              if (h_index_mapping(jspec, mapped_iz, mapped_ix) != -1) {
                h_index_mapping(ispec, iz, ix) =
                    h_index_mapping(jspec, mapped_iz, mapped_ix);
                previously_assigned = true;
                break;
              }
            }
          }
        }
        if (!previously_assigned) {
          const auto [ix, iz] =
              element_connections.coordinates_at_corner(corner);
          h_index_mapping(ispec, iz, ix) = iglob;
          iglob++;
        }
      }
    }
  }

  Kokkos::deep_copy(index_mapping, h_index_mapping);
}
