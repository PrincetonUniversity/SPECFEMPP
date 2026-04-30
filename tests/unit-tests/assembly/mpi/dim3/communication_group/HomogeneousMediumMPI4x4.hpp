#pragma once

#include "expected_groups.hpp"

namespace TestData::CommunicationGroup::HomogeneousMediumMPI4x4 {

constexpr static int ngll = 5;

using namespace specfem::assembly_test;

const static ExpectedFaceCommunicationGroup expected_group_0_1 = {
  0,    // my_rank
  1,    // neighbor_rank
  2,    // nfaces
  ngll, // ngll
  {
      // Per-face data for (Include some or all faces in the group)
      // Face 0
      {
          specfem::mesh_entity::dim3::type::right, // my_orientation
          specfem::mesh_entity::dim3::type::left,  // neighbor_orientation
          0,                                       // theta
          1,                                       // my_element
          0                                        // neighbor_element
      }
      // Add more faces as needed
  }
};

const static ExpectedFaceCommunicationGroup expected_group_0_2 = {
  0,    // my_rank
  2,    // neighbor_rank
  2,    // nfaces
  ngll, // ngll
  {
      // Per-face data for (Include some or all faces in the group)
      // Face 0
      {
          specfem::mesh_entity::dim3::type::back,  // my_orientation
          specfem::mesh_entity::dim3::type::front, // neighbor_orientation
          0,                                       // theta
          2,                                       // my_element
          0                                        // neighbor_element
      }
      // Add more faces as needed
  }
};

const static ExpectedFaceCommunicationGroup expected_group_1_0 = {
  1,    // my_rank
  3,    // neighbor_rank
  2,    // nfaces
  ngll, // ngll
  {
      // Per-face data for (Include some or all faces in the group)
      // Face 0
      {
          specfem::mesh_entity::dim3::type::back,  // my_orientation
          specfem::mesh_entity::dim3::type::front, // neighbor_orientation
          0,                                       // theta
          3,                                       // my_element
          1                                        // neighbor_element
      }
      // Add more faces as needed
  }
};

const static ExpectedEdgeCommunicationGroup expected_edge_group_0_3 = {
  0,    // my_rank
  3,    // neighbor_rank
  1,    // nedges
  ngll, // ngll
  {
      // Per-edge data for (Include some or all edges in the group)
      // Edge 0
      {
          specfem::mesh_entity::dim3::type::back_right, // my_orientation
          specfem::mesh_entity::dim3::type::front_left, // neighbor_orientation
          0,                                            // reflect
          3,                                            // my_element
          0                                             // neighbor_element
      }
      // Add more edges as needed
  }
};

const static auto expected = std::make_tuple(
    ExpectedMPIFaceCommunicationGroups{ expected_group_0_1, expected_group_0_2,
                                        expected_group_1_0 },
    ExpectedMPIEdgeCommunicationGroups{ expected_edge_group_0_3 },
    ExpectedMPICornerCommunicationGroups{} // No corner groups in this test case
);

} // namespace TestData::CommunicationGroup::HomogeneousMediumMPI4x4
