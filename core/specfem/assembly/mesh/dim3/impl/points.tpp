#include "specfem/assembly/mesh.hpp"

template <typename CoordinateView, typename ShapeFunctionView,
          typename ControlNodeCoordinates>
void initialize_coordinates(
    const int nspec, const int ngllz, const int nglly, const int ngllx,
    const int ngnod, const CoordinateView &coordinates,
    const ShapeFunctionView &shape_functions,
    const ControlNodeCoordinates &control_node_coordinates) {

  Kokkos::parallel_for(
      "specfem::assembly::mesh::points::initialize_coordinates",
      Kokkos::MDRangePolicy<Kokkos::Rank<4> >({ 0, 0, 0, 0 },
                                              { nspec, ngllz, nglly, ngllx }),
      KOKKOS_LAMBDA(const int ispec, const int iz, const int iy, const int ix) {
        for (int ia = 0; ia < ngnod; ia++) {
          coordinates(ispec, iz, iy, ix, 0) +=
              shape_functions(iz, iy, ix, ia) *
              control_node_coordinates(ispec, ia, 0);
          coordinates(ispec, iz, iy, ix, 1) +=
              shape_functions(iz, iy, ix, ia) *
              control_node_coordinates(ispec, ia, 1);
          coordinates(ispec, iz, iy, ix, 2) +=
              shape_functions(iz, iy, ix, ia) *
              control_node_coordinates(ispec, ia, 2);
        }
      });

  Kokkos::fence();
}

specfem::assembly::mesh_impl::points<specfem::element::dimension_tag::dim3>::points(
    const int &nspec, const int &ngllz, const int &nglly, const int &ngllx,
    const specfem::mesh::adjacency_graph<dimension_tag>
        &adjacency_graph,
    const specfem::assembly::mesh_impl::control_nodes<dimension_tag>
        &control_nodes,
    const specfem::assembly::mesh_impl::shape_functions<dimension_tag>
        &shape_functions)
    : nspec(nspec), ngllz(ngllz), nglly(nglly), ngllx(ngllx),
      index_mapping("specfem::assembly::mesh::points::index_mapping", nspec,
                    ngllz, nglly, ngllx),
      coord("specfem::assembly::mesh::points::coord", nspec, ngllz, nglly,
            ngllx, ndim),
      h_index_mapping(Kokkos::create_mirror_view(index_mapping)),
      h_coord(Kokkos::create_mirror_view(coord)) {

  // Use adjacency graph to set up quadrature point indexing
  const auto &graph = adjacency_graph.graph();

  constexpr int chunk_size = 1;
  const int nchunks = nspec / chunk_size + (nspec % chunk_size == 0 ? 0 : 1);

  // Initialize to -1
  Kokkos::parallel_for(
      "specfem::assembly::mesh::points::initialize_index_mapping",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<4> >({ 0, 0, 0, 0 },
                                              { nspec, ngllz, nglly, ngllx }),
      [=](const int ispec, const int iz, const int iy, const int ix) {
        this->h_index_mapping(ispec, iz, iy, ix) = -1;
      });

  Kokkos::fence();

  // Set up internal points
  Kokkos::parallel_for(
      "specfem::assembly::mesh::points::initialize_internal_indices",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<4> >(
          { 0, 1, 1, 1 }, { nchunks, ngllz - 1, nglly - 1, ngllx - 1 }),
      [=](const int ichunk, const int iz, const int iy, const int ix) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk * chunk_size + ielement;
          if (ispec >= nspec)
            break;
          this->h_index_mapping(ispec, iz, iy, ix) =
              ielement + iz * chunk_size + iy * chunk_size * (ngllz - 2) +
              ix * chunk_size * (ngllz - 2) * (nglly - 2) +
              ichunk * chunk_size * (ngllz - 2) * (nglly - 2) * (ngllx - 2);
        }
      });

  int ig = nspec * (ngllz - 2) * (nglly - 2) * (ngllx - 2);

  // Filter out strongly conforming connections
  auto filter = [&graph](const auto &edge) {
    return graph[edge].connection ==
           specfem::element_connections::type::strongly_conforming;
  };

  // Create a filtered graph view
  const auto fg = boost::make_filtered_graph(graph, filter);

  const auto mapping = specfem::mesh_entity::element(ngllz, nglly, ngllx);

  // Iterate over all faces
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    // Iterate over all faces
    for (auto iface : specfem::mesh_entity::dim3::faces) {
      const int npoints = mapping.number_of_points_on_orientation(iface);
      for (auto ipoint = 0; ipoint < npoints; ipoint++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          const auto [iz, iy, ix] = mapping.map_coordinates(iface, ipoint);
          // skip points on edges and corners
          if (((iz == 0 || iz == ngllz - 1) &&
               (iy == 0 || iy == nglly - 1 || ix == 0 || ix == ngllx - 1)) ||
              ((iy == 0 || iy == nglly - 1) &&
               (ix == 0 || ix == ngllx - 1 || iz == 0 || iz == ngllz - 1)) ||
              ((ix == 0 || ix == ngllx - 1) &&
               (iz == 0 || iz == ngllz - 1 || iy == 0 || iy == nglly - 1)))
            continue;

          // Check if already assigned
          bool previously_assigned = false;
          for (auto face :
               boost::make_iterator_range(boost::out_edges(ispec, fg))) {
            if (fg[face].orientation == iface) {
              const int jspec = boost::target(face, fg);
              const auto other_face = boost::edge(jspec, ispec, graph).first;
              const auto jface = fg[other_face].orientation;

              const auto connections = specfem::element_connections::connection_mapping(
                  ngllz, nglly, ngllx,
                  Kokkos::subview(control_nodes.h_control_node_index, ispec,
                                  Kokkos::ALL),
                  Kokkos::subview(control_nodes.h_control_node_index, jspec,
                                  Kokkos::ALL));

              const auto [mapped_iz, mapped_iy, mapped_ix] =
                  connections.map_coordinates(iface, jface, iz, iy, ix);

              if (this->h_index_mapping(jspec, mapped_iz, mapped_iy,
                                        mapped_ix) != -1) {
                this->h_index_mapping(ispec, iz, iy, ix) =
                    this->h_index_mapping(jspec, mapped_iz, mapped_iy,
                                          mapped_ix);
                previously_assigned = true;
                break;
              }
            }
          }
          if (!previously_assigned) {
            this->h_index_mapping(ispec, iz, iy, ix) = ig;
            ig++;
          }
        }
      }
    }

    // Iterate over all edges
    for (auto iedge : specfem::mesh_entity::dim3::edges) {
      const int npoints = mapping.number_of_points_on_orientation(iedge);
      for (int ipoint = 0; ipoint < npoints; ipoint++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          const auto [iz, iy, ix] = mapping.map_coordinates(iedge, ipoint);
          // skip points on corners
          if ((iz == 0 || iz == ngllz - 1) && (iy == 0 || iy == nglly - 1) &&
              (ix == 0 || ix == ngllx - 1))
            continue;

          auto valid_connections = specfem::mesh_entity::faces_of_edge(iedge);
          valid_connections.push_back(iedge);
          bool previously_assigned = false;
          for (auto edge :
               boost::make_iterator_range(boost::out_edges(ispec, fg))) {
            if (specfem::mesh_entity::contains(valid_connections,
                                               fg[edge].orientation)) {
              const int jspec = boost::target(edge, fg);
              const auto other_face = boost::edge(jspec, ispec, graph).first;
              const auto jorientation = fg[other_face].orientation;
              const auto connections = specfem::element_connections::connection_mapping(
                  ngllz, nglly, ngllx,
                  Kokkos::subview(control_nodes.h_control_node_index, ispec,
                                  Kokkos::ALL),
                  Kokkos::subview(control_nodes.h_control_node_index, jspec,
                                  Kokkos::ALL));

              const auto [mapped_iz, mapped_iy, mapped_ix] =
                  connections.map_coordinates(fg[edge].orientation,
                                              jorientation, iz, iy, ix);
              if (this->h_index_mapping(jspec, mapped_iz, mapped_iy,
                                        mapped_ix) != -1) {
                this->h_index_mapping(ispec, iz, iy, ix) =
                    this->h_index_mapping(jspec, mapped_iz, mapped_iy,
                                          mapped_ix);
                previously_assigned = true;
                break;
              }
            }
          }
          if (!previously_assigned) {
            this->h_index_mapping(ispec, iz, iy, ix) = ig;
            ig++;
          }
        }
      }
    }

    // Finally we need to treat corner points
    for (auto icorner : specfem::mesh_entity::dim3::corners) {
      for (int ielement = 0; ielement < chunk_size; ielement++) {
        int ispec = ichunk + ielement;
        if (ispec >= nspec)
          break;
        const auto [iz, iy, ix] = mapping.map_coordinates(icorner);

        auto valid_faces = specfem::mesh_entity::faces_of_corner(icorner);
        auto valid_connections = specfem::mesh_entity::edges_of_corner(icorner);

        // append valid_faces to valid_connections
        valid_connections.insert(valid_connections.end(), valid_faces.begin(),
                                 valid_faces.end());
        valid_connections.push_back(icorner);

        // Check if already assigned
        bool previously_assigned = false;
        for (auto corner :
             boost::make_iterator_range(boost::out_edges(ispec, fg))) {
          if (specfem::mesh_entity::contains(valid_connections,
                                             fg[corner].orientation)) {
            const int jspec = boost::target(corner, fg);
            const auto other_face = boost::edge(jspec, ispec, graph).first;
            const auto jorientation = fg[other_face].orientation;
            const auto connections = specfem::element_connections::connection_mapping(
                ngllz, nglly, ngllx,
                Kokkos::subview(control_nodes.h_control_node_index, ispec,
                                Kokkos::ALL),
                Kokkos::subview(control_nodes.h_control_node_index, jspec,
                                Kokkos::ALL));

            int mapped_iz, mapped_iy, mapped_ix;
            if (specfem::mesh_entity::contains(
                    specfem::mesh_entity::dim3::corners,
                    fg[corner].orientation)) {
              std::tie(mapped_iz, mapped_iy, mapped_ix) =
                  connections.map_coordinates(fg[corner].orientation,
                                              jorientation);
            } else {
              std::tie(mapped_iz, mapped_iy, mapped_ix) =
                  connections.map_coordinates(fg[corner].orientation,
                                              jorientation, iz, iy, ix);
            }
            if (this->h_index_mapping(jspec, mapped_iz, mapped_iy, mapped_ix) !=
                -1) {
              this->h_index_mapping(ispec, iz, iy, ix) =
                  this->h_index_mapping(jspec, mapped_iz, mapped_iy, mapped_ix);
              previously_assigned = true;
              break;
            }
          }
        }
        if (!previously_assigned) {
          this->h_index_mapping(ispec, iz, iy, ix) = ig;
          ig++;
        }
      }
    }
  }

  this->nglob = ig;

  // Make sure all points have been assigned
  Kokkos::parallel_for(
      "specfem::assembly::mesh::points::check_all_points_assigned",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<4> >({ 0, 0, 0, 0 },
                                              { nspec, ngllz, nglly, ngllx }),
      [=](const int ispec, const int iz, const int iy, const int ix) {
        if (this->h_index_mapping(ispec, iz, iy, ix) == -1) {
          std::stringstream ss;
          ss << "Point not assigned for element " << ispec << " at (" << iz
             << ", " << iy << ", " << ix << ").";
          throw std::runtime_error(ss.str());
        }
      });

  Kokkos::fence();

  Kokkos::deep_copy(this->index_mapping, this->h_index_mapping);

  this->coord = CoordViewType("specfem::assembly::mesh::points::coord", nspec,
                              ngllz, nglly, ngllx, ndim);
  this->h_coord = Kokkos::create_mirror_view(this->coord);

  const int ngnod = control_nodes.ngnod;

  initialize_coordinates(nspec, ngllz, nglly, ngllx, ngnod, this->coord,
                         shape_functions.shape3D,
                         control_nodes.control_node_coordinates);

  Kokkos::deep_copy(this->h_coord, this->coord);

  return;
}
