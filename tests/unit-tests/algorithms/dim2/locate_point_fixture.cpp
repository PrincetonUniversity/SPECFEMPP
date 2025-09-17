#include "locate_point_fixture.hpp"

std::vector<std::vector<std::pair<double, double> > >
LocatePoint2D::create_unit_square(int ngll, double xmin, double xmax,
                                  double zmin, double zmax) {
  std::vector<std::pair<double, double> > coords;
  for (int iz = 0; iz < ngll; iz++) {
    for (int ix = 0; ix < ngll; ix++) {
      double x = xmin + (xmax - xmin) * ix / (ngll - 1);
      double z = zmin + (zmax - zmin) * iz / (ngll - 1);
      coords.push_back({ x, z });
    }
  }
  return { coords };
}
std::vector<std::vector<std::pair<double, double> > >
LocatePoint2D::create_control_nodes(int ngnod, double xmin, double xmax,
                                    double zmin, double zmax) {
  std::vector<std::pair<double, double> > nodes;

  if (ngnod == 4) {
    // 4-node quadrilateral: corners only
    nodes = {
      { xmin, zmin }, // Bottom-left
      { xmax, zmin }, // Bottom-right
      { xmax, zmax }, // Top-right
      { xmin, zmax }  // Top-left
    };
  } else if (ngnod == 9) {
    // 9-node quadrilateral: corners + edge midpoints + center
    double xmid = (xmin + xmax) / 2.0;
    double zmid = (zmin + zmax) / 2.0;
    nodes = {
      { xmin, zmin }, // 0: Bottom-left corner
      { xmax, zmin }, // 1: Bottom-right corner
      { xmax, zmax }, // 2: Top-right corner
      { xmin, zmax }, // 3: Top-left corner
      { xmid, zmin }, // 4: Bottom edge midpoint
      { xmax, zmid }, // 5: Right edge midpoint
      { xmid, zmax }, // 6: Top edge midpoint
      { xmin, zmid }, // 7: Left edge midpoint
      { xmid, zmid }  // 8: Center
    };
  } else {
    throw std::runtime_error("Unsupported ngnod: " + std::to_string(ngnod));
  }

  return { nodes };
}
HostView4d LocatePoint2D::create_coordinate_array(
    const std::vector<std::vector<std::pair<double, double> > >
        &element_coords) {
  int nspec = element_coords.size();
  int ngll = std::sqrt(element_coords[0].size());

  HostView4d coords("coords", nspec, ngll, ngll, 2);

  for (int ispec = 0; ispec < nspec; ispec++) {
    int idx = 0;
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        coords(ispec, iz, ix, 0) = element_coords[ispec][idx].first;  // x
        coords(ispec, iz, ix, 1) = element_coords[ispec][idx].second; // z
        idx++;
      }
    }
  }
  return coords;
}
LocatePoint2D::ElementGeometry LocatePoint2D::create_element_geometry(
    const std::vector<std::vector<std::pair<double, double> > > &element_coords,
    const std::vector<std::vector<std::pair<double, double> > >
        &control_coords) {

  ElementGeometry geom;
  geom.nspec = element_coords.size();
  geom.ngllx = std::sqrt(element_coords[0].size());
  geom.ngllz = geom.ngllx;
  geom.ngnod = control_coords[0].size();
  int ngllxz = geom.ngllx * geom.ngllz;

  // Create coordinate array
  auto coords_double = create_coordinate_array(element_coords);

  // Use tested utility functions to replicate assign_numbering logic
  auto points = flatten_coordinates(coords_double);
  auto sorted_points = points;
  sort_points_spatially(sorted_points);
  type_real tolerance =
      compute_spatial_tolerance(sorted_points, geom.nspec, ngllxz);
  int nglob = assign_global_numbering(sorted_points, tolerance);
  auto reordered_points = reorder_to_original_layout(sorted_points);
  auto bbox = compute_bounding_box(reordered_points);

  // Use the create_coordinate_arrays function to get proper index_mapping and
  // coordinates
  auto [index_mapping, global_coords, nglob_actual] =
      create_coordinate_arrays(reordered_points, geom.nspec, geom.ngllx, nglob);

  // Assign to geometry structure
  geom.index_mapping = index_mapping;
  geom.global_coords = global_coords;
  geom.nglob = nglob_actual;

  // Set up control nodes
  geom.control_nodes =
      Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>(
          "control_nodes", 2, geom.nspec, geom.ngnod);
  for (int ispec = 0; ispec < geom.nspec; ispec++) {
    for (int inode = 0; inode < geom.ngnod; inode++) {
      geom.control_nodes(0, ispec, inode) = control_coords[ispec][inode].first;
      geom.control_nodes(1, ispec, inode) = control_coords[ispec][inode].second;
    }
  }

  // Set bounding box
  geom.xmin = bbox.xmin;
  geom.xmax = bbox.xmax;
  geom.zmin = bbox.zmin;
  geom.zmax = bbox.zmax;

  return geom;
}

// Create a single unit square element [0,1] x [0,1] with 2x2 GLL points
LocatePoint2D::ElementGeometry LocatePoint2D::create_single_unit_square_2x2() {
  auto element_coords =
      LocatePoint2D::create_unit_square(2, 0.0, 1.0, 0.0, 1.0);
  auto control_coords =
      LocatePoint2D::create_control_nodes(4, 0.0, 1.0, 0.0, 1.0);
  return LocatePoint2D::create_element_geometry(element_coords, control_coords);
}
// Create two adjacent elements [0,0.5]x[0,0.5] and [0.5,1]x[0,0.5] with 2x2
// GLL points
LocatePoint2D::ElementGeometry
LocatePoint2D::create_two_adjacent_elements_2x2() {
  std::vector<std::vector<std::pair<double, double> > > element_coords = {
    LocatePoint2D::create_unit_square(2, 0.0, 0.5, 0.0, 0.5)[0], // Left element
    LocatePoint2D::create_unit_square(2, 0.5, 1.0, 0.0, 0.5)[0] // Right element
                                                                // (shares edge
                                                                // at x=0.5)
  };
  auto control_coords = std::vector<std::vector<std::pair<double, double> > >{
    LocatePoint2D::create_control_nodes(4, 0.0, 0.5, 0.0, 0.5)[0], // Left
                                                                   // element
    LocatePoint2D::create_control_nodes(4, 0.5, 1.0, 0.0, 0.5)[0]  // Right
                                                                   // element
  };
  return LocatePoint2D::create_element_geometry(element_coords, control_coords);
}
// Create 2x2 grid of elements with 2x2 GLL points each
LocatePoint2D::ElementGeometry LocatePoint2D::create_2x2_grid_elements_2x2() {
  std::vector<std::vector<std::pair<double, double> > > element_coords = {
    LocatePoint2D::create_unit_square(2, 0.0, 1.0, 0.0, 1.0)[0], // Element 0:
                                                                 // bottom-left
    LocatePoint2D::create_unit_square(2, 1.0, 2.0, 0.0, 1.0)[0], // Element 1:
                                                                 // bottom-right
    LocatePoint2D::create_unit_square(2, 0.0, 1.0, 1.0, 2.0)[0], // Element 2:
                                                                 // top-left
    LocatePoint2D::create_unit_square(2, 1.0, 2.0, 1.0, 2.0)[0]  // Element 3:
                                                                 // top-right
  };
  auto control_coords = std::vector<std::vector<std::pair<double, double> > >{
    LocatePoint2D::create_control_nodes(4, 0.0, 1.0, 0.0, 1.0)[0], // Element 0
    LocatePoint2D::create_control_nodes(4, 1.0, 2.0, 0.0, 1.0)[0], // Element 1
    LocatePoint2D::create_control_nodes(4, 0.0, 1.0, 1.0, 2.0)[0], // Element 2
    LocatePoint2D::create_control_nodes(4, 1.0, 2.0, 1.0, 2.0)[0]  // Element 3
  };
  return LocatePoint2D::create_element_geometry(element_coords, control_coords);
}
// Create a single unit square element [0,1] x [0,1] with 5x5 GLL points
LocatePoint2D::ElementGeometry LocatePoint2D::create_single_unit_square_5x5() {
  auto element_coords =
      LocatePoint2D::create_unit_square(5, 0.0, 1.0, 0.0, 1.0);
  auto control_coords =
      LocatePoint2D::create_control_nodes(4, 0.0, 1.0, 0.0, 1.0);
  return LocatePoint2D::create_element_geometry(element_coords, control_coords);
}
// Create a single unit square element [0,1] x [0,1] with 2x2 GLL points and 9
// control nodes
LocatePoint2D::ElementGeometry
LocatePoint2D::create_single_unit_square_2x2_9node() {
  auto element_coords =
      LocatePoint2D::create_unit_square(2, 0.0, 1.0, 0.0, 1.0);
  auto control_coords =
      LocatePoint2D::create_control_nodes(9, 0.0, 1.0, 0.0, 1.0);
  return LocatePoint2D::create_element_geometry(element_coords, control_coords);
}

// Create a single unit square element [0,1] x [0,1] with 5x5 GLL points and 9
// control nodes
LocatePoint2D::ElementGeometry
LocatePoint2D::create_single_unit_square_5x5_9node() {
  auto element_coords =
      LocatePoint2D::create_unit_square(5, 0.0, 1.0, 0.0, 1.0);
  auto control_coords =
      LocatePoint2D::create_control_nodes(9, 0.0, 1.0, 0.0, 1.0);
  return LocatePoint2D::create_element_geometry(element_coords, control_coords);
}

// Create two adjacent elements with 5x5 GLL points
LocatePoint2D::ElementGeometry
LocatePoint2D::create_two_adjacent_elements_5x5() {
  std::vector<std::vector<std::pair<double, double> > > element_coords = {
    LocatePoint2D::create_unit_square(5, 0.0, 0.5, 0.0, 0.5)[0], // Left element
    LocatePoint2D::create_unit_square(5, 0.5, 1.0, 0.0, 0.5)[0] // Right element
                                                                // (shares edge
                                                                // at x=0.5)
  };
  auto control_coords = std::vector<std::vector<std::pair<double, double> > >{
    LocatePoint2D::create_control_nodes(4, 0.0, 0.5, 0.0, 0.5)[0], // Left
                                                                   // element
    LocatePoint2D::create_control_nodes(4, 0.5, 1.0, 0.0, 0.5)[0]  // Right
                                                                   // element
  };
  return LocatePoint2D::create_element_geometry(element_coords, control_coords);
}
