#pragma once

#include "enumerations/connections.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include <boost/graph/filtered_graph.hpp>

void specfem::assembly::mesh<specfem::dimension::type::dim2>::assemble(
    const specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>
        &graph) {

  const int nspec = this->nspec;
  const int ngllx = this->ngllx;
  const int ngllz = this->ngllz;

  const int ngnod = this->ngnod;

  const auto xi = this->h_xi;
  const auto gamma = this->h_xi;

  const auto shape2D = this->h_shape2D;
  const auto coorg = this->h_control_node_coord;

  this->xmin = std::numeric_limits<type_real>::max();
  this->xmax = std::numeric_limits<type_real>::min();
  this->zmin = std::numeric_limits<type_real>::max();
  this->zmax = std::numeric_limits<type_real>::min();

  // Get the coordinates for each quadrature point
  // ----
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> ***,
      Kokkos::DefaultHostExecutionSpace>
      global_coordinates("specfem::assembly::mesh::h_coord", nspec, ngllz,
                         ngllx);

  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        auto shape_functions =
            Kokkos::subview(h_shape2D, iz, ix, Kokkos::ALL());

        double xcor = 0.0;
        double zcor = 0.0;

        for (int in = 0; in < ngnod; in++) {
          xcor += coorg(0, ispec, in) * shape_functions[in];
          zcor += coorg(1, ispec, in) * shape_functions[in];
        }

        global_coordinates(ispec, iz, ix).x = xcor;
        global_coordinates(ispec, iz, ix).z = zcor;

        if (this->xmin > xcor)
          this->xmin = xcor;
        if (this->zmin > zcor)
          this->zmin = zcor;
        if (this->xmax < xcor)
          this->xmax = xcor;
        if (this->zmax < zcor)
          this->zmax = zcor;
      }
    }
  }

  const type_real tolerance = std::min(xmax - xmin, zmax - zmin) * 1e-6;
  // ----

  constexpr int chunk_size = specfem::parallel_config::storage_chunk_size;

  this->h_index_mapping = Kokkos::View<int ***, Kokkos::LayoutLeft,
                                       Kokkos::DefaultHostExecutionSpace>(
      "specfem::assembly::mesh::h_index_mapping", nspec, ngllz, ngllx);
  this->h_coord = Kokkos::View<type_real ****, Kokkos::LayoutRight,
                               Kokkos::DefaultHostExecutionSpace>(
      "specfem::assembly::mesh::h_coord", ndim, nspec, ngllz, ngllx);

  // Initialize index mapping and coordinates
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        this->h_index_mapping(ispec, iz, ix) = -1;
      }
    }
  }

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
          this->h_index_mapping(ispec, iz, ix) = iglob;
          this->h_coord(0, ispec, iz, ix) = global_coordinates(ispec, iz, ix).x;
          this->h_coord(1, ispec, iz, ix) = global_coordinates(ispec, iz, ix).z;
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
      for (int ipoint = 1; ipoint < npoints - 1;
           ipoint++) { // we loop only over interior points of edge
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

              // Get the correct coordinates for the edge point on the other
              // edge
              const auto [from_coords, to_coords] =
                  element_connections.map_coordinates(iedge, mapped_iedge,
                                                      ipoint);
              const auto [mapped_iz, mapped_ix] = to_coords;

              if (this->h_index_mapping(jspec, mapped_iz, mapped_ix) != -1) {

                this->h_index_mapping(ispec, iz, ix) =
                    this->h_index_mapping(jspec, mapped_iz, mapped_ix);
                this->h_coord(0, ispec, iz, ix) =
                    global_coordinates(ispec, iz, ix).x;
                this->h_coord(1, ispec, iz, ix) =
                    global_coordinates(ispec, iz, ix).z;
                previously_assigned = true;
                break;
              }
            }
          }
          // ----

          // If the point has not been assigned yet, we assign it a new index
          // and store the coordinates
          if (!previously_assigned) {
            this->h_index_mapping(ispec, iz, ix) = iglob;
            this->h_coord(0, ispec, iz, ix) =
                global_coordinates(ispec, iz, ix).x;
            this->h_coord(1, ispec, iz, ix) =
                global_coordinates(ispec, iz, ix).z;
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

        // Get the edges that are connected to this corner
        auto valid_connections = specfem::mesh_entity::edges_of_corner(corner);

        // We also need to add the corner itself to the valid connections
        // This is necessary to handle cases where a corner connection is made
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

            // Check if the connection is an edge connection
            if (specfem::mesh_entity::contains(specfem::mesh_entity::edges,
                                               fg[edge].orientation)) {
              const auto point = element_connections.find_corner_on_edge(
                  corner, fg[edge].orientation);
              const auto [from_coords, to_coords] =
                  element_connections.map_coordinates(fg[edge].orientation,
                                                      mapped_iedge, point);
              const auto [mapped_iz, mapped_ix] = to_coords;

              if (this->h_index_mapping(jspec, mapped_iz, mapped_ix) != -1) {
                this->h_index_mapping(ispec, iz, ix) =
                    this->h_index_mapping(jspec, mapped_iz, mapped_ix);
                this->h_coord(0, ispec, iz, ix) =
                    global_coordinates(ispec, iz, ix).x;
                this->h_coord(1, ispec, iz, ix) =
                    global_coordinates(ispec, iz, ix).z;
                previously_assigned = true;
                break;
              }
            } else {
              // Corner to corner connection
              const auto [from_coords, to_coords] =
                  element_connections.map_coordinates(corner, mapped_iedge, 0);
              const auto [mapped_iz, mapped_ix] = to_coords;

              if (this->h_index_mapping(jspec, mapped_iz, mapped_ix) != -1) {
                this->h_index_mapping(ispec, iz, ix) =
                    this->h_index_mapping(jspec, mapped_iz, mapped_ix);
                this->h_coord(0, ispec, iz, ix) =
                    global_coordinates(ispec, iz, ix).x;
                this->h_coord(1, ispec, iz, ix) =
                    global_coordinates(ispec, iz, ix).z;
                previously_assigned = true;
                break;
              }
            }
          }
        }
        if (!previously_assigned) {
          this->h_index_mapping(ispec, iz, ix) = iglob;
          this->h_coord(0, ispec, iz, ix) = global_coordinates(ispec, iz, ix).x;
          this->h_coord(1, ispec, iz, ix) = global_coordinates(ispec, iz, ix).z;
          iglob++;
        }
      }
    }
  }

  this->index_mapping =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>(
          "specfem::assembly::mesh::index_mapping", nspec, ngllz, ngllx);
  this->coord = Kokkos::View<type_real ****, Kokkos::LayoutRight,
                             Kokkos::DefaultExecutionSpace>(
      "specfem::assembly::mesh::coord", ndim, nspec, ngllz, ngllx);

  // Copy the host views to device views
  Kokkos::deep_copy(this->index_mapping, this->h_index_mapping);
  Kokkos::deep_copy(this->coord, this->h_coord);

  // Check all points have been assigned a global index
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        if (this->h_index_mapping(ispec, iz, ix) == -1) {
          std::ostringstream oss;
          oss << "Error: Point (" << ispec << ", " << iz << ", " << ix
              << ") has not been assigned a global index.";
          throw std::runtime_error(oss.str());
        }
      }
    }
  }

  this->nglob = iglob;
}
