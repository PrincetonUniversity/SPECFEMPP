#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "IO/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "mesh/modifiers/modifiers.hpp"
#include <limits>
#include <stdexcept>

namespace specfem {
namespace test {
namespace mesh_modifiers {
namespace subdivisions {

// we may want to look into implementing a kd tree. Instead, just bin things
// into a quadtree, using heap-stored nodes
template <typename T, int num_children> struct tree_node {
  tree_node<T, num_children> *parent;
  T data;
  std::vector<tree_node<T, num_children> > children;
  int depth;
  bool leaf;

  tree_node() : depth(-1), leaf(true) {}

  void gen_children() {
    if (leaf) {
      children = std::vector<tree_node<T, num_children> >(num_children);
      for (int i = 0; i < num_children; i++) {
        children[i] = tree_node<T, num_children>();
        children[i].parent = this;
        children[i].depth = depth + 1;
      }
      leaf = false;
    }
  }
};

// utility to store control nodes. contains interpolation functions
struct elem_coords {
  type_real pts[3][3][2];
  type_real cx, cz;
  unsigned char mode;

  elem_coords() = default;
  elem_coords(const Kokkos::View<int **, Kokkos::HostSpace> &knods,
              const Kokkos::View<type_real **, Kokkos::HostSpace> &coord,
              const int ispec) {
    const int ngnod = knods.extent(0);
    switch (ngnod) {
    case 4: {
      for (int idim = 0; idim < 2; idim++) {
        pts[0][0][idim] = coord(idim, knods(0, ispec));
        pts[0][1][idim] = coord(idim, knods(1, ispec));
        pts[1][1][idim] = coord(idim, knods(2, ispec));
        pts[1][0][idim] = coord(idim, knods(3, ispec));
      }
      mode = 0;
      std::tie(cx, cz) = interpolate(0, 0);
      return;
    }
    case 9: {
      for (int idim = 0; idim < 2; idim++) {
        pts[0][0][idim] = coord(idim, knods(0, ispec));
        pts[0][2][idim] = coord(idim, knods(1, ispec));
        pts[2][2][idim] = coord(idim, knods(2, ispec));
        pts[2][0][idim] = coord(idim, knods(3, ispec));
        pts[0][1][idim] = coord(idim, knods(4, ispec));
        pts[1][2][idim] = coord(idim, knods(5, ispec));
        pts[2][1][idim] = coord(idim, knods(6, ispec));
        pts[1][0][idim] = coord(idim, knods(7, ispec));
        pts[1][1][idim] = coord(idim, knods(8, ispec));
      }
      mode = 1;
      cx = pts[1][1][0];
      cz = pts[1][1][1];
      return;
    }
    default:
      throw std::runtime_error(
          "number of mesh control nodes invalid: ngnod = " +
          std::to_string(ngnod));
      return;
    }
  }
  elem_coords(const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
              const int ispec)
      : elem_coords(mesh.control_nodes.knods, mesh.control_nodes.coord, ispec) {
  }

  std::pair<type_real, type_real> interpolate(type_real xi, type_real gamma) {
    switch (mode) {
    case 0: {
      type_real bot, top;
      type_real out[2];
      for (int idim = 0; idim < 2; idim++) {
        bot = (pts[0][0][idim] * (1 - xi) + pts[0][1][idim] * (1 + xi)) / 2;
        top = (pts[1][0][idim] * (1 - xi) + pts[1][1][idim] * (1 + xi)) / 2;
        out[idim] = (bot * (1 - gamma) + top * (1 + gamma)) / 2;
      }
      return std::make_pair(out[0], out[1]);
    }
    case 1: {
      type_real xavgs[3];
      type_real out[2];
      type_real xi2 = xi * xi;
      type_real ga2 = gamma * gamma;
      type_real coefxi1 = (xi2 - xi) / 2;
      type_real coefxi2 = (1 - xi2);
      type_real coefxi3 = (xi + xi2) / 2;

      type_real coefga1 = (ga2 - gamma) / 2;
      type_real coefga2 = (1 - ga2);
      type_real coefga3 = (gamma + ga2) / 2;
      for (int idim = 0; idim < 2; idim++) {
        for (int iz = 0; iz < 3; iz++) {
          xavgs[iz] = pts[iz][0][idim] * coefxi1 + pts[iz][1][idim] * coefxi2 +
                      pts[iz][2][idim] * coefxi3;
        }
        out[idim] =
            xavgs[0] * coefga1 + xavgs[1] * coefga2 + xavgs[2] * coefga3;
      }
      return std::make_pair(out[0], out[1]);
    }
    default:
      return std::make_pair(-1, -1);
    }
  }
  type_real small_len2() {
    type_real x1, z1, x2, z2;
    std::tie(x1, z1) = interpolate(-1, 0);
    std::tie(x2, z2) = interpolate(1, 0);
    type_real len = (x2 - x1) * (x2 - x1) + (z2 - z1) * (z2 - z1);
    std::tie(x1, z1) = interpolate(0, -1);
    std::tie(x2, z2) = interpolate(0, 1);
    return std::min(len, (x2 - x1) * (x2 - x1) + (z2 - z1) * (z2 - z1));
  }
};

// bins to store elements into.
struct elem_coord_bin2D {
  std::vector<int> indices; // this will only be nonempty for leaf nodes
  type_real xmin, xmax, ymin, ymax, cx, cy; // coordinates for bin (aabb)
  tree_node<elem_coord_bin2D, 4> *parent;   // parent->data == *this

  elem_coord_bin2D() = default;
  elem_coord_bin2D(tree_node<elem_coord_bin2D, 4> &treenode, type_real xmin,
                   type_real xmax, type_real ymin, type_real ymax)
      : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), parent(&treenode),
        cx((xmin + xmax) / 2), cy((ymin + ymax) / 2) {}
  elem_coord_bin2D(tree_node<elem_coord_bin2D, 4> &treenode, int index)
      : parent(&treenode) {
    tree_node<elem_coord_bin2D, 4> &parent = *treenode.parent;
    type_real cx = (parent.data.xmax + parent.data.xmin) / 2;
    type_real cy = (parent.data.ymax + parent.data.ymin) / 2;
    // inherit aabb from quad-subdivided parent's aabb
    switch (index) {
    case 0:
      // lower left
      xmin = parent.data.xmin;
      ymin = parent.data.ymin;
      xmax = cx;
      ymax = cy;
      break;
    case 1:
      // lower right
      xmin = cx;
      ymin = parent.data.ymin;
      xmax = parent.data.xmax;
      ymax = cy;
      break;
    case 2:
      // upper left
      xmin = parent.data.xmin;
      ymin = cy;
      xmax = cx;
      ymax = parent.data.ymax;
      break;
    case 3:
      // upper right
      xmin = cx;
      ymin = cy;
      xmax = parent.data.ymax;
      ymax = parent.data.ymax;
      break;
    }
    this->cx = (xmin + xmax) / 2;
    this->cy = (ymin + ymax) / 2;
  }

  /**
   * @brief Subdivides this node. every index in this node is then passed to one
   * of the children.
   *
   * @param coordref - ispec -> elemdata mapping
   */
  void subdivide(const std::vector<elem_coords> &coordref) {
    if (!parent->leaf) {
      // subdivide underlying tree structure
      parent->gen_children();
      for (int i = 0; i < 4; i++) {
        parent->children[i].data = elem_coord_bin2D(parent->children[i], i);
      }
    }

    // insert() will properly insert each index into the correct child
    for (const int &i : indices) {
      insert(coordref, i);
    }
    // we're moving, not copying
    indices.clear();
  }

  /**
   * @brief Inserts the ispec into the tree at this node. If we are a leaf,
   * places into this index. Otherwise, it is inserted into the correct child.
   *
   * @param coordref - ispec -> elemdata mapping
   * @param ispec - element to store
   * @param max_storage - threshold for when a leaf is large enough to warrant
   * subdivision.
   * @param maxdepth - the furthest subdivision allowed before max_storage is
   * ignored.
   */
  void insert(const std::vector<elem_coords> &coordref, int ispec,
              int max_storage = 20, int maxdepth = 10) {
    if (parent->leaf || parent->depth >= maxdepth) {
      indices.push_back(ispec);
      return;
    } else {
      const int sub = ((coordref[ispec].cx > cx) ? 1 : 0) |
                      ((coordref[ispec].cz > cy) ? 2 : 0);
      auto &child = parent->children[sub].data;
      if (child.indices.size() >= max_storage) {
        child.subdivide(coordref);
      }
      child.insert(coordref, ispec, max_storage, maxdepth);
    }
  }

  /**
   * @brief Finds the k nearest elements to the given coordinates.
   *
   * @param coordref - element coord vector
   * @param x - x coordinate to find elements near
   * @param y - y (z) coordinate to find elements near
   * @param k - number of elements to search for
   * @param nearest_inds a vector of <ispec, dist^2> for the solutions. This
   * should start empty, but will be set, in order from closest to farthest.
   */
  void knearest(const std::vector<elem_coords> &coordref, const type_real x,
                const type_real y, const int k,
                std::vector<std::pair<int, type_real> > &nearest_inds) {
    // trace down to leaf containing (x,y), unless we collected everything, and
    // we're too far

    int ncollected = nearest_inds.size();
    if (ncollected >= k) {
      // nearest_inds is full. return if the this box cannot beat the last of
      // the closest
      type_real boxclosestx = std::min(xmax, std::max(xmin, x));
      type_real boxclosesty = std::min(ymax, std::max(ymin, y));
      type_real dist2 = (x - boxclosestx) * (x - boxclosestx) +
                        (y - boxclosesty) * (y - boxclosesty);
      if (dist2 > nearest_inds[k - 1].second) {
        return;
      }
    }
    if (parent->leaf) {
      // linsearch
      int nelems = indices.size();

      type_real maxdist = (ncollected >= k)
                              ? nearest_inds[k - 1].second
                              : std::numeric_limits<type_real>::max();
      for (int ielem = 0; ielem < nelems; ielem++) {
        int ispec = indices[ielem];
        type_real dist = (x - coordref[ispec].cx) * (x - coordref[ispec].cx) +
                         (y - coordref[ispec].cz) * (y - coordref[ispec].cz);
        if (dist < maxdist) {
          if (ncollected < k) {
            ncollected++;
            // the last entry will always be overwritten
            nearest_inds.push_back(std::make_pair(0, 0));
          }

          // make it so ibubble is the open space where the collecteds are in
          // order.
          int ibubble = ncollected - 1;
          while (ibubble > 0 && nearest_inds[ibubble - 1].second > dist) {
            nearest_inds[ibubble] = nearest_inds[ibubble - 1];
            ibubble--;
          }
          nearest_inds[ibubble] = std::make_pair(ispec, dist);
          // update maxdist in case we need to
          maxdist = (ncollected >= k) ? nearest_inds[k - 1].second
                                      : std::numeric_limits<type_real>::max();
        }
      }
    } else {
      // try the "closest" sub-boxes in order. no complicated math; just simple
      // heuristics
      int xprio = (x > cx) ? 1 : 0;
      int yprio = (y > cy) ? 2 : 0;
      parent->children[yprio + xprio].data.knearest(coordref, x, y, k,
                                                    nearest_inds);
      parent->children[(2 - yprio) + xprio].data.knearest(coordref, x, y, k,
                                                          nearest_inds);
      parent->children[yprio + (1 - xprio)].data.knearest(coordref, x, y, k,
                                                          nearest_inds);
      parent->children[(2 - yprio) + (1 - xprio)].data.knearest(
          coordref, x, y, k, nearest_inds);
    }
  }
};

void test_mesh_unifsubdivisions2D(
    specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    int global_subdiv_factor_x, int global_subdiv_factor_z) {
  const int ngnod = mesh.control_nodes.ngnod;

  type_real xmin, xmax, zmin, zmax;
  xmin = xmax = mesh.control_nodes.coord(0, 0);
  zmin = zmax = mesh.control_nodes.coord(1, 0);

  for (int ispec = 0; ispec < mesh.nspec; ispec++) {
    xmin = std::min(xmin, mesh.control_nodes.coord(0, ispec));
    zmin = std::min(zmin, mesh.control_nodes.coord(1, ispec));
    xmax = std::max(xmax, mesh.control_nodes.coord(0, ispec));
    zmax = std::max(zmax, mesh.control_nodes.coord(1, ispec));
  }

  specfem::mesh::mesh<specfem::dimension::type::dim2> meshcopy = mesh;
  specfem::mesh::modifiers<specfem::dimension::type::dim2> modifiers;
  for (int imat = 0; imat < mesh.materials.n_materials; imat++) {
    modifiers.set_subdivision(
        imat, { global_subdiv_factor_x, global_subdiv_factor_z });
  }
  modifiers.apply(meshcopy);

  // check meshes
  specfem::kokkos::HostView3d<bool> hits("test_mesh_unifsubdivisions2D hits",
                                         mesh.nspec, global_subdiv_factor_z,
                                         global_subdiv_factor_x);
  // storing control nodes
  std::vector<elem_coords> base_elems;
  // quadtree bins
  tree_node<elem_coord_bin2D, 4> base_bins;
  base_bins.data = elem_coord_bin2D(base_bins, xmin, xmax, zmin, zmax);
  base_bins.depth = 0;
  for (int ispec = 0; ispec < mesh.nspec; ispec++) {
    base_elems.push_back(elem_coords(mesh, ispec));
    base_bins.data.insert(base_elems, ispec);
  }

  for (int ispec = 0; ispec < meshcopy.nspec; ispec++) {
    elem_coords elem(meshcopy, ispec);
    // find the tuple (ispec, isubx, isubz) that this ispec_new should be in
    std::vector<std::pair<int, type_real> > nearest_inds;
    constexpr int num_nearest_search = 9;
    base_bins.data.knearest(base_elems, elem.cx, elem.cz, num_nearest_search,
                            nearest_inds);
    int ispec_near, isubx_near, isubz_near;
    type_real min_err = std::numeric_limits<type_real>::max();
    type_real equality_threshold = elem.small_len2() * 1e-3;
    type_real inv_subfact_x = 1.0 / global_subdiv_factor_x;
    type_real inv_subfact_z = 1.0 / global_subdiv_factor_z;
    for (int inear = 0; inear < num_nearest_search; inear++) {
      int ispec_orig = nearest_inds[inear].first;
      elem_coords &elem_orig = base_elems[ispec_orig];
      // the center of elem *should* have parameter 2*[ (isub + 0.5)/sub ] - 1
      for (int isubx = 0; isubx < global_subdiv_factor_x; isubx++) {
        for (int isubz = 0; isubz < global_subdiv_factor_z; isubz++) {
          type_real x, z;
          std::tie(x, z) =
              elem_orig.interpolate((2 * isubx + 1) * inv_subfact_x - 1,
                                    (2 * isubz + 1) * inv_subfact_z - 1);
          type_real err =
              (x - elem.cx) * (x - elem.cx) + (z - elem.cz) * (z - elem.cz);
          if (err < min_err) {
            min_err = err;
            ispec_near = ispec_orig;
            isubx_near = isubx;
            isubz_near = isubz;
          }
        }
      }
      if (min_err < equality_threshold) {
        // we can be sure we found it.
        break;
      }
    }
    // now check: are the control nodes in good places?
    type_real pt_errs[3][3];
    elem_coords &elem_near = base_elems[ispec_near];
    type_real xi[3] = { (2 * isubx_near) * inv_subfact_x - 1,
                        (2 * isubx_near + 1) * inv_subfact_x - 1,
                        (2 * isubx_near + 2) * inv_subfact_x - 1 };
    type_real ga[3] = { (2 * isubz_near) * inv_subfact_z - 1,
                        (2 * isubz_near + 1) * inv_subfact_z - 1,
                        (2 * isubz_near + 2) * inv_subfact_z - 1 };
    bool fail = false;
    for (int ix = 0; ix < 3; ix++) {
      for (int iz = 0; iz < 3; iz++) {
        type_real x1, z1, x2, z2;
        std::tie(x1, z1) = elem.interpolate(ix - 1, iz - 1);
        std::tie(x2, z2) = elem_near.interpolate(xi[ix], ga[iz]);
        pt_errs[ix][iz] = (x2 - x1) * (x2 - x1) + (z2 - z1) * (z2 - z1);
        if (pt_errs[ix][iz] > equality_threshold) {
          fail = true;
        }
      }
    }

    if (fail) {
      char pt_char[3][3];
      for (int ix = 0; ix < 3; ix++) {
        for (int iz = 0; iz < 3; iz++) {
          pt_char[ix][iz] = (pt_errs[ix][iz] > equality_threshold) ? 'x' : ' ';
        }
      }

      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " - global subdivision: " << global_subdiv_factor_x << "x"
             << global_subdiv_factor_z << "\n"
             << " - subdivided element (" << isubx_near << "," << isubz_near
             << ") of element " << ispec_near << "\n"
             << " - 9 node chart:\n"
             << "     " << pt_char[0][2] << pt_char[1][2] << pt_char[2][2]
             << "\n"
             << "     " << pt_char[0][1] << pt_char[1][1] << pt_char[2][1]
             << "\n"
             << "     " << pt_char[0][0] << pt_char[1][0] << pt_char[2][0]
             << "\n\n"
             << "     " << pt_errs[0][2] << "\t" << pt_errs[1][2] << "\t"
             << pt_errs[2][2] << "\n"
             << "     " << pt_errs[0][1] << "\t" << pt_errs[1][1] << "\t"
             << pt_errs[2][1] << "\n"
             << "     " << pt_errs[0][0] << "\t" << pt_errs[1][0] << "\t"
             << pt_errs[2][0] << "\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
    // this is precisely the subdivision. Ensure the material is correct
    if (mesh.materials.material_index_mapping(ispec_near).database_index !=
        meshcopy.materials.material_index_mapping(ispec).database_index) {
      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " - global subdivision: " << global_subdiv_factor_x << "x"
             << global_subdiv_factor_z << "\n"
             << " - subdivided element (" << isubx_near << "," << isubz_near
             << ") of element " << ispec_near << "\n"
             << " - Incorrect material index mapping:\n"
             << "     should be "
             << mesh.materials.material_index_mapping(ispec_near).database_index
             << "\n"
             << "     found "
             << meshcopy.materials.material_index_mapping(ispec).database_index
             << "\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
    if (hits(ispec_near, isubz_near, isubx_near)) {
      // we already hit this element; something went wrong.
      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " - global subdivision: " << global_subdiv_factor_x << "x"
             << global_subdiv_factor_z << "\n"
             << " - subdivided element (" << isubx_near << "," << isubz_near
             << ") of element " << ispec_near << "\n"
             << " - This subelement was hit multiple times!\n"
             << "     second ispec_new that hit: " << ispec << "\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
    // mark this element as hit
    hits(ispec_near, isubz_near, isubx_near) = true;
  }

  // ensure every element was hit
  for (int ispec = 0; ispec < mesh.nspec; ispec++) {
    for (int isubz = 0; isubz < global_subdiv_factor_z; isubz++) {
      for (int isubx = 0; isubx < global_subdiv_factor_x; isubx++) {
        if (!hits(ispec, isubz, isubx)) {
          FAIL() << "--------------------------------------------------\n"
                 << "\033[0;31m[FAILED]\033[0m Test failed\n"
                 << " - global subdivision: " << global_subdiv_factor_x << "x"
                 << global_subdiv_factor_z << "\n"
                 << " - subdivided element (" << isubx << "," << isubz
                 << ") of element " << ispec << "\n"
                 << " - This subelement was never hit!\n"
                 << "--------------------------------------------------\n\n"
                 << std::endl;
        }
      }
    }
  }
}
} // namespace subdivisions
} // namespace mesh_modifiers
} // namespace test
} // namespace specfem

TEST(MESH_MODIFIERS, subdivision) {
  auto mpi = MPIEnvironment::get_mpi();
  auto mesh =
      specfem::IO::read_mesh("../../../tests/unit-tests/displacement_tests/"
                             "Newmark/serial/test1/database.bin",
                             mpi);

  specfem::test::mesh_modifiers::subdivisions::test_mesh_unifsubdivisions2D(
      mesh, 2, 3);
  specfem::test::mesh_modifiers::subdivisions::test_mesh_unifsubdivisions2D(
      mesh, 4, 2);
}
