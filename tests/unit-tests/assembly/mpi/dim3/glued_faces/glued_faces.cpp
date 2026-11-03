// Cross-rank face pairing for every way two right-handed hexahedra can be
// glued.
//
// Two ranks. Element A (rank 0) is the unit cube. Element B (rank 1) sits
// directly below A, placed by one of the 24 proper rotations of the cube, so
// the face of B touching A's bottom face ranges over all six face types in all
// four in-plane orientations -- including the "flipped" case where B's bottom
// face is glued to A's bottom face. Each case is also run with each of the
// four shared corners as the anchor point.
//
// Each rank also holds a second, unrotated element glued on the far side (A2
// on top of A, B2 under B) so that every element has a same-rank neighbor;
// the cross-rank interface is still exactly the one A-B face.
//
// For every case, the GLL coordinates each rank packs for the neighbor must
// coincide with the coordinates at the slot the neighbor unpacks them into.

#include "SPECFEM_Environment.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/mpi.hpp"
#include "specfem/quadrature.hpp"
#include <algorithm>
#include <array>
#include <boost/graph/adjacency_list.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace glued_faces_test_impl {

constexpr auto dimension = specfem::element::dimension_tag::dim3;
constexpr auto elastic = specfem::element::medium_tag::elastic;
constexpr int ngll = 5;

using Vec = std::array<double, 3>;
using Mat = std::array<std::array<int, 3>, 3>;

/** @brief Elements of the two-rank mesh. */
enum class Element { a, a2, b, b2 };

/** @brief Reference position (x, y, z) in {0, 1}^3 of hex8 control node k. */
Vec reference_node(const int node) {
  constexpr int xyz[8][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 },
                              { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 },
                              { 1, 1, 1 }, { 0, 1, 1 } };
  return { static_cast<double>(xyz[node][0]), static_cast<double>(xyz[node][1]),
           static_cast<double>(xyz[node][2]) };
}

/** @brief The 24 rotations of the cube (signed permutations, det = +1). */
std::vector<Mat> proper_rotations() {
  std::vector<Mat> rotations;
  const int perms[6][3] = { { 0, 1, 2 }, { 0, 2, 1 }, { 1, 0, 2 },
                            { 1, 2, 0 }, { 2, 0, 1 }, { 2, 1, 0 } };
  for (const auto &p : perms) {
    for (int signs = 0; signs < 8; ++signs) {
      Mat m{};
      for (int row = 0; row < 3; ++row) {
        m[row][p[row]] = (signs >> row) & 1 ? -1 : 1;
      }
      const int det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
      if (det == 1)
        rotations.push_back(m);
    }
  }
  return rotations;
}

/** @brief Physical position of control node @p node of @p element. */
Vec physical_node(const Element element, const Mat &rotation, const int node) {
  Vec r = reference_node(node);
  switch (element) {
  case Element::a:
    return r;
  case Element::a2:
    r[2] += 1.0;
    return r;
  case Element::b2:
    r[2] -= 2.0;
    return r;
  case Element::b:
    break;
  }
  // B: rotate about the cube center, then shift one unit down in z.
  Vec p{};
  for (int row = 0; row < 3; ++row) {
    double v = 0.0;
    for (int col = 0; col < 3; ++col)
      v += rotation[row][col] * (r[col] - 0.5);
    p[row] = v + 0.5;
  }
  p[2] -= 1.0;
  return p;
}

/** @brief Corner entity whose single control node is @p node. */
specfem::mesh_entity::dim3::type corner_of_node(const int node) {
  for (const auto corner : specfem::mesh_entity::dim3::corners) {
    if (specfem::mesh_entity::nodes_on_orientation(corner)[0] == node)
      return corner;
  }
  throw std::runtime_error("no corner for node");
}

/** @brief Face of @p element whose four nodes all satisfy z == z0. */
specfem::mesh_entity::dim3::type
face_at_z(const Element element, const Mat &rotation, const double z0) {
  for (const auto face : specfem::mesh_entity::dim3::faces) {
    bool all = true;
    for (const int node : specfem::mesh_entity::nodes_on_orientation(face)) {
      all = all &&
            std::abs(physical_node(element, rotation, node)[2] - z0) < 1e-12;
    }
    if (all)
      return face;
  }
  throw std::runtime_error("no face at z0");
}

/** @brief Whether two points coincide. */
bool same_point(const Vec &p, const Vec &q) {
  return std::abs(p[0] - q[0]) + std::abs(p[1] - q[1]) + std::abs(p[2] - q[2]) <
         1e-12;
}

/** @brief One rank's two-element mesh and its MPI interfaces. */
struct Case {
  specfem::mesh::cartesian3d_mesh mesh;
  specfem::assembly::mesh<dimension> assembly_mesh;
  specfem::assembly::element_types<dimension> element_types;
  specfem::assembly::fields<dimension> fields;
  specfem::assembly::mpi<dimension> mpi;

  Case(const Mat &rotation, const int anchor_node_a) {
    const bool is_b = specfem::MPI::get_rank() == 1;
    // Mesh order: the interface element first, its same-rank neighbor second.
    const std::array<Element, 2> elements =
        is_b ? std::array<Element, 2>{ Element::b, Element::b2 }
             : std::array<Element, 2>{ Element::a, Element::a2 };

    mesh.nspec = 2;
    mesh.element_grid = specfem::mesh_entity::element<dimension>(ngll);

    // Control nodes, shared between the two elements where they coincide.
    std::vector<Vec> positions;
    std::array<std::array<int, 8>, 2> index{};
    for (int e = 0; e < 2; ++e) {
      for (int node = 0; node < 8; ++node) {
        const Vec p = physical_node(elements[e], rotation, node);
        const auto it =
            std::find_if(positions.begin(), positions.end(),
                         [&p](const Vec &q) { return same_point(p, q); });
        index[e][node] = static_cast<int>(it - positions.begin());
        if (it == positions.end())
          positions.push_back(p);
      }
    }

    auto &nodes = mesh.control_nodes;
    nodes = specfem::mesh::control_nodes<dimension>(
        8, static_cast<int>(positions.size()));
    nodes.nspec = 2;
    nodes.control_node_index =
        specfem::mesh::control_nodes<dimension>::ControlNodeIndexViewType(
            "control_node_index", 2, 8);
    double lo[3] = { 1e9, 1e9, 1e9 };
    double hi[3] = { -1e9, -1e9, -1e9 };
    for (std::size_t q = 0; q < positions.size(); ++q) {
      for (int c = 0; c < 3; ++c) {
        nodes.coordinates(q, c) = positions[q][c];
        lo[c] = std::min(lo[c], positions[q][c]);
        hi[c] = std::max(hi[c], positions[q][c]);
      }
    }
    for (int e = 0; e < 2; ++e)
      for (int node = 0; node < 8; ++node)
        nodes.control_node_index(e, node) = index[e][node];
    nodes.xmin = lo[0];
    nodes.xmax = hi[0];
    nodes.ymin = lo[1];
    nodes.ymax = hi[1];
    nodes.zmin = lo[2];
    nodes.zmax = hi[2];

    // Same-rank glue: A top <-> A2 bottom, or B's lower face <-> B2 top.
    mesh.adjacency_graph =
        specfem::mesh::adjacency_graph<dimension>(mesh.nspec);
    {
      using EdgeProperties =
          specfem::mesh::adjacency_graph<dimension>::EdgeProperties;
      const auto conforming =
          specfem::element_connections::type::strongly_conforming;
      const auto inner_face = is_b ? face_at_z(Element::b, rotation, -1.0)
                                   : specfem::mesh_entity::dim3::type::top;
      const auto outer_face = is_b ? specfem::mesh_entity::dim3::type::top
                                   : specfem::mesh_entity::dim3::type::bottom;
      auto &graph = mesh.adjacency_graph.local_connections();
      boost::add_edge(0, 1, EdgeProperties(conforming, inner_face), graph);
      boost::add_edge(1, 0, EdgeProperties(conforming, outer_face), graph);
    }

    // The cross-rank face and the anchor corner, identified on both sides by
    // the same physical point.
    const auto face_a = face_at_z(Element::a, rotation, 0.0);
    const auto face_b = face_at_z(Element::b, rotation, 0.0);
    const Vec anchor_point = physical_node(Element::a, rotation, anchor_node_a);
    int anchor_node_b = -1;
    for (int node = 0; node < 8; ++node) {
      if (same_point(physical_node(Element::b, rotation, node), anchor_point))
        anchor_node_b = node;
    }
    const auto anchor_a = corner_of_node(anchor_node_a);
    const auto anchor_b = corner_of_node(anchor_node_b);

    mesh.adjacency_graph.mpi_connections().emplace_back(
        specfem::element_connections::type::strongly_conforming,
        is_b ? face_b : face_a, static_cast<std::size_t>(is_b ? 0 : 1),
        is_b ? face_a : face_b, std::size_t{ 0 }, std::size_t{ 0 },
        is_b ? anchor_b : anchor_a, is_b ? anchor_a : anchor_b);

    mesh.tags.nspec = 2;
    mesh.tags.tags_container =
        Kokkos::View<specfem::mesh::impl::tags_container *, Kokkos::HostSpace>(
            "tags", 2);
    for (int e = 0; e < 2; ++e) {
      mesh.tags.tags_container(e) = specfem::mesh::impl::tags_container(
          elastic, specfem::element::property_tag::isotropic,
          specfem::element::attenuation_tag::none,
          specfem::element::boundary_tag::none,
          e == 0 ? specfem::element::mpi_tag::outer
                 : specfem::element::mpi_tag::inner);
    }

    const specfem::quadrature::quadratures quadratures(
        specfem::quadrature::gll::gll{});
    assembly_mesh = { mesh.nspec,
                      nodes.ngnod,
                      ngll,
                      ngll,
                      ngll,
                      mesh.tags,
                      mesh.adjacency_graph,
                      mesh.control_nodes,
                      quadratures };
    element_types = { mesh.nspec, assembly_mesh.element_grid, assembly_mesh,
                      mesh.tags };
    fields = { assembly_mesh, element_types,
               specfem::simulation::type::forward };
    mpi = { mesh.adjacency_graph,
            element_types,
            specfem::simulation::type::forward,
            fields,
            assembly_mesh.h_mesh_to_compute,
            ngll,
            ngll,
            ngll };
  }

  /**
   * @brief Number of exchanged points that land on a different physical
   * point on the receiving rank (0 when the face pairing is right).
   */
  int mispaired_points() const {
    const auto &field = fields.template get_simulation_field<
        specfem::simulation::field_type::forward>();
    std::vector<Vec> xyz(field.template get_nglob<elastic>());
    for (int ispec = 0; ispec < mesh.nspec; ++ispec)
      for (int iz = 0; iz < ngll; ++iz)
        for (int iy = 0; iy < ngll; ++iy)
          for (int ix = 0; ix < ngll; ++ix)
            for (int c = 0; c < 3; ++c)
              xyz[field.template get_iglob<false, elastic>(ispec, iz, iy, ix)]
                 [c] = assembly_mesh.h_coord(ispec, iz, iy, ix, c);

    const auto &patterns =
        mpi.forward_communication_patterns
            .template get<specfem::tags::Tags<dimension, elastic>>();
    if (patterns.size() != 1)
      return -1;
    const auto &[neighbor, pattern] = *patterns.begin();

    std::vector<double> send, recv(3 * pattern.unpack.nglob);
    for (unsigned int i = 0; i < pattern.pack.nglob; ++i) {
      const auto &x = xyz[pattern.pack.h_mapping(i)];
      send.insert(send.end(), x.begin(), x.end());
    }
    SPECFEM_MPI_SAFECALL(MPI_Sendrecv(
        send.data(), static_cast<int>(send.size()), MPI_DOUBLE,
        static_cast<int>(neighbor), 900, recv.data(),
        static_cast<int>(recv.size()), MPI_DOUBLE, static_cast<int>(neighbor),
        900, specfem::MPI::communicator(), MPI_STATUS_IGNORE));

    int wrong = 0;
    for (unsigned int i = 0; i < pattern.unpack.nglob; ++i) {
      const auto &x = xyz[pattern.unpack.h_mapping(i)];
      wrong +=
          !(std::abs(x[0] - recv[3 * i]) + std::abs(x[1] - recv[3 * i + 1]) +
                std::abs(x[2] - recv[3 * i + 2]) <
            1e-5);
    }
    return wrong;
  }
};

} // namespace glued_faces_test_impl

TEST(GluedFacesMPI3D, EveryGluePairsPoints) {
  if (!SPECFEMEnvironment::IsMPISizeValid()) {
    GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();
  }
  namespace impl = glued_faces_test_impl;
  const auto rotations = impl::proper_rotations();
  ASSERT_EQ(rotations.size(), 24u);

  // (B face glued to A's bottom) -> {cases, cases with mispaired points}
  std::map<std::string, std::array<int, 2>> summary;

  for (std::size_t r = 0; r < rotations.size(); ++r) {
    // The four nodes of A's bottom face are the candidate anchors.
    for (const int anchor_node_a : specfem::mesh_entity::nodes_on_orientation(
             specfem::mesh_entity::dim3::type::bottom)) {
      const impl::Case c(rotations[r], anchor_node_a);
      int wrong = c.mispaired_points();
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &wrong, 1, MPI_INT,
                                         MPI_MAX,
                                         specfem::MPI::communicator()));
      const std::string key =
          "bottom<->" + specfem::mesh_entity::dim3::to_string(impl::face_at_z(
                            impl::Element::b, rotations[r], 0.0));
      auto &entry = summary[key];
      entry[0] += 1;
      entry[1] += wrong != 0;
      EXPECT_EQ(wrong, 0) << key << " rotation " << r << " anchor node "
                          << anchor_node_a << ": " << wrong
                          << " mispaired points";
    }
  }

  if (specfem::MPI::get_rank() == 0) {
    std::ostringstream os;
    os << "\nface pair         cases  mispaired\n";
    for (const auto &[key, entry] : summary) {
      os << key << std::string(18 - key.size(), ' ') << entry[0] << "     "
         << entry[1] << "\n";
    }
    std::cout << os.str() << std::flush;
  }
}

TEST(GluedFacesMPI3D, FaceGridHandednessMatchesGeometry) {
  const specfem::mesh_entity::element<glued_faces_test_impl::dimension> element(
      glued_faces_test_impl::ngll);
  const int n = glued_faces_test_impl::ngll;
  for (const auto face : specfem::mesh_entity::dim3::faces) {
    const auto xyz = [&](const int point) {
      const auto [iz, iy, ix] = element.map_coordinates(face, point);
      return std::array<int, 3>{ ix, iy, iz };
    };
    const auto o = xyz(0), di = xyz(1), dj = xyz(n);
    std::array<int, 3> ei, ej;
    for (int k = 0; k < 3; ++k) {
      ei[k] = di[k] - o[k];
      ej[k] = dj[k] - o[k];
    }
    const std::array<int, 3> cross = { ei[1] * ej[2] - ei[2] * ej[1],
                                       ei[2] * ej[0] - ei[0] * ej[2],
                                       ei[0] * ej[1] - ei[1] * ej[0] };
    int expected = 0;
    for (int k = 0; k < 3; ++k) {
      if (cross[k] != 0)
        expected = (o[k] == n - 1 ? 1 : -1) * (cross[k] > 0 ? 1 : -1);
    }
    EXPECT_EQ(specfem::mesh_entity::dim3::face_grid_handedness(face), expected)
        << specfem::mesh_entity::dim3::to_string(face);
  }
}
