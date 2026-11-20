#include "../test_fixture.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/subview.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/enums.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Assembly3DTest run on meshes that include a coupled acoustic-elastic
 * mesh, so at least one medium's DOFs start above 0.
 *
 * Kept apart from the shared Assembly3DTests instantiation, whose tests assume
 * the eight-element constant-density boxes.
 */
class SimulationField3DTest : public Assembly3DTest {};

INSTANTIATE_TEST_SUITE_P(SimulationField3DTests, SimulationField3DTest,
                         ::testing::Values("EightNodeElastic",
                                           "EightNodeElasticCosserat",
                                           "AcousticElasticInterface"));

TEST_P(SimulationField3DTest, DofNumbering) {
  constexpr auto dim3 = specfem::element::dimension_tag::dim3;
  using FieldType = specfem::assembly::simulation_field<
      dim3, specfem::simulation::field_type::forward>;

  const auto &assembly = getAssembly();
  const FieldType field(assembly.mesh, assembly.element_types);

  const int nspec = field.nspec;
  const int ngllz = field.ngllz;
  const int nglly = field.nglly;
  const int ngllx = field.ngllx;

  const auto &mesh_points = assembly.mesh.h_index_mapping;

  int total_nglob = 0;
  int media_present = 0;
  bool element_offset_exercised = false;

  specfem::tag_dispatch::for_each(
      FieldType::combinations, [&]<typename TagsType>() {
        constexpr auto medium = TagsType::medium_tag;
        const auto medium_name = specfem::element::to_string(medium);
        const int nglob = field.template get_nglob<medium>();
        total_nglob += nglob;

        // The medium's table holds exactly its own elements.
        const auto medium_elements =
            assembly.element_types.get_elements_on_host(medium);
        const auto local = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            field.local_index_mapping.template get<TagsType>());
        ASSERT_EQ(static_cast<int>(local.extent(0)), medium_elements.size())
            << medium_name;

        if (nglob == 0)
          return;
        ++media_present;
        if (medium_elements.begin_index() > 0)
          element_offset_exercised = true;

        // Every entry is a row of the medium's arrays, and every row is used.
        std::vector<int> hits(nglob, 0);
        int out_of_range = 0;
        for (std::size_t i = 0; i < local.size(); ++i) {
          const int row = local.data()[i];
          if (row < 0 || row >= nglob)
            ++out_of_range;
          else
            ++hits[row];
        }
        EXPECT_EQ(out_of_range, 0) << medium_name;
        EXPECT_EQ(std::count(hits.begin(), hits.end(), 0), 0)
            << medium_name << ": local DOFs never referenced";

        // Points share a row exactly when they share a mesh point, and
        // get_iglob (indexed by mesh element) reads the same row.
        const int first = medium_elements.begin_index();
        std::vector<int> row_to_mesh_point(nglob, -1);
        int inconsistent = 0;
        int get_iglob_mismatches = 0;
        for (int ispec = first; ispec < medium_elements.end_index(); ++ispec)
          for (int iz = 0; iz < ngllz; ++iz)
            for (int iy = 0; iy < nglly; ++iy)
              for (int ix = 0; ix < ngllx; ++ix) {
                const int row = local(ispec - first, iz, iy, ix);
                if (row !=
                    field.template get_iglob<false, medium>(ispec, iz, iy, ix))
                  ++get_iglob_mismatches;
                if (row < 0 || row >= nglob)
                  continue;
                const int mesh_point = mesh_points(ispec, iz, iy, ix);
                if (row_to_mesh_point[row] == -1)
                  row_to_mesh_point[row] = mesh_point;
                else if (row_to_mesh_point[row] != mesh_point)
                  ++inconsistent;
              }
        EXPECT_EQ(get_iglob_mismatches, 0) << medium_name;
        EXPECT_EQ(inconsistent, 0) << medium_name;
        // Distinct rows must be distinct mesh points.
        std::vector<int> mesh_points_seen = row_to_mesh_point;
        std::sort(mesh_points_seen.begin(), mesh_points_seen.end());
        EXPECT_EQ(std::adjacent_find(mesh_points_seen.begin(),
                                     mesh_points_seen.end()),
                  mesh_points_seen.end())
            << medium_name << ": two rows share a mesh point";

        // A subview of a range starting inside the medium is the table sliced
        // at the range's offset within the medium.
        if (medium_elements.size() > 1) {
          const int offset = 1;
          const specfem::datatype::TaggedElementIndexRange<
              specfem::tags::Tags<dim3, medium>>
              elements(specfem::datatype::ElementIndexRange(
                  first + offset, medium_elements.end_index()));
          const auto sub = specfem::assembly::subview(field, elements);
          const auto h_sub = Kokkos::create_mirror_view_and_copy(
              Kokkos::HostSpace(), sub.index_mapping);
          ASSERT_EQ(static_cast<int>(h_sub.extent(0)), elements.size())
              << medium_name;
          int subview_mismatches = 0;
          for (int e = 0; e < elements.size(); ++e)
            for (int iz = 0; iz < ngllz; ++iz)
              for (int iy = 0; iy < nglly; ++iy)
                for (int ix = 0; ix < ngllx; ++ix)
                  if (h_sub(e, iz, iy, ix) != local(offset + e, iz, iy, ix))
                    ++subview_mismatches;
          EXPECT_EQ(subview_mismatches, 0) << medium_name;
        }

        // A range reaching outside the medium's elements is rejected.
        if (medium_elements.size() < nspec) {
          const specfem::datatype::TaggedElementIndexRange<
              specfem::tags::Tags<dim3, medium>>
              all_elements(specfem::datatype::ElementIndexRange(0, nspec));
          EXPECT_THROW(specfem::assembly::subview(field, all_elements),
                       std::out_of_range)
              << medium_name;
        }
      });

  EXPECT_EQ(total_nglob, field.nglob);

  // A coupled mesh must actually exercise a medium starting past element 0.
  if (media_present > 1)
    EXPECT_TRUE(element_offset_exercised);
  if (GetParam() == std::string("AcousticElasticInterface"))
    EXPECT_GT(media_present, 1);
}
