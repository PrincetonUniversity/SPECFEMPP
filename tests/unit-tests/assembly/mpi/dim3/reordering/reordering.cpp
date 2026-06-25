/**
 * @file reordering.cpp
 * @brief Unit tests for the compute-domain element reordering invariant in 3D
 * spectral element assembly.
 *
 * specfem::assembly::mesh_impl::mesh_to_compute_mapping reorders mesh elements
 * into a compute ordering that groups elements by
 * (medium, property, boundary) and, within each such group, places the inner
 * (partition-interior) elements before the outer (MPI-boundary) elements. This
 * keeps each (medium, property, boundary, mpi_tag) sublist a contiguous
 * compute-index range, which the SIMD stiffness kernels rely on (lane L reads
 * mesh element `ispec + L`).
 *
 * These tests validate, on a partitioned mesh, that:
 *  1. mesh_to_compute / compute_to_mesh form a valid bijection (permutation).
 *  2. Within each (medium, property, boundary) group, no inner element appears
 *     after an outer element, and each group occupies a single contiguous run.
 *
 * The outer classification is derived independently from the adjacency graph's
 * MPI connections (mirroring the production classifier), so the test does not
 * depend on element_types having been built with MPI tags.
 *
 * @see specfem::assembly::mesh_impl::mesh_to_compute_mapping
 */

#include "../fixture.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/mpi.hpp"
#include <set>
#include <tuple>
#include <vector>

/**
 * @brief Validate the mesh<->compute index mapping is a bijection.
 */
TEST_P(AssemblyMPI3DTest, ComputeMeshBijection) {
  const auto &assembly_mesh = getAssemblyMesh();
  const int nspec = assembly_mesh.nspec;
  const auto &h_mesh_to_compute = assembly_mesh.h_mesh_to_compute;
  const auto &h_compute_to_mesh = assembly_mesh.h_compute_to_mesh;

  std::vector<bool> compute_seen(nspec, false);
  for (int m = 0; m < nspec; ++m) {
    const int c = h_mesh_to_compute(m);
    ASSERT_GE(c, 0) << "compute index out of range for mesh element " << m;
    ASSERT_LT(c, nspec) << "compute index out of range for mesh element " << m;
    ASSERT_FALSE(compute_seen[c])
        << "compute index " << c << " assigned to more than one mesh element";
    compute_seen[c] = true;
    ASSERT_EQ(h_compute_to_mesh(c), m)
        << "compute_to_mesh is not the inverse of mesh_to_compute at mesh "
           "element "
        << m;
  }
}

/**
 * @brief Validate inner-before-outer contiguity within each element-type group.
 */
TEST_P(AssemblyMPI3DTest, InnerBeforeOuterPerGroup) {
  const auto &assembly_mesh = getAssemblyMesh();
  const auto &mesh = getMesh();
  const int nspec = assembly_mesh.nspec;
  const auto &h_compute_to_mesh = assembly_mesh.h_compute_to_mesh;

  // Derive outer classification (mesh ordering) from the MPI connections,
  // mirroring the production classifier in mesh_to_compute_mapping.
  std::vector<bool> is_outer(nspec, false);
  for (const auto &conn : mesh.adjacency_graph.mpi_connections()) {
    is_outer[static_cast<int>(conn.local_index)] = true;
  }

  using GroupKey = std::tuple<int, int, int>; // (medium, property, boundary)
  const auto key_of = [&](int mesh_ispec) -> GroupKey {
    const auto tag = mesh.tags.tags_container(mesh_ispec);
    return { static_cast<int>(tag.medium_tag),
             static_cast<int>(tag.property_tag),
             static_cast<int>(tag.boundary_tag) };
  };

  std::set<GroupKey> completed_groups;
  GroupKey current_key{};
  bool have_current = false;
  bool saw_outer = false;

  for (int c = 0; c < nspec; ++c) {
    const int mesh_ispec = h_compute_to_mesh(c);
    const GroupKey key = key_of(mesh_ispec);

    if (!have_current || key != current_key) {
      // Starting a new contiguous run: the previous group is complete.
      if (have_current) {
        completed_groups.insert(current_key);
      }
      ASSERT_EQ(completed_groups.count(key), 0u)
          << "group (medium=" << std::get<0>(key)
          << ", property=" << std::get<1>(key)
          << ", boundary=" << std::get<2>(key)
          << ") is not contiguous in compute ordering (reappears at compute "
             "index "
          << c << ")";
      current_key = key;
      have_current = true;
      saw_outer = false;
    }

    if (is_outer[mesh_ispec]) {
      saw_outer = true;
    } else {
      ASSERT_FALSE(saw_outer)
          << "inner element (compute index " << c << ", mesh element "
          << mesh_ispec << ") appears after an outer element within group "
          << "(medium=" << std::get<0>(current_key)
          << ", property=" << std::get<1>(current_key)
          << ", boundary=" << std::get<2>(current_key) << ")";
    }
  }
}
