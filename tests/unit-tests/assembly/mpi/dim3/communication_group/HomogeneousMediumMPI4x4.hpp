#pragma once

#include "specfem/mesh_entity.hpp"

namespace TestData::CommunicationGroup::HomogeneousMediumMPI4x4 {

constexpr static int ngll = 5;

using namespace specfem::assembly_test;

// Expected communication group for process 0
// -- (neighbor_rank=1) --
const static auto expected_group_0_1 = ExpectedCommunicationGroup{
  .my_rank = 0,
  .neighbor_rank = 1,
  .nfaces = 4,
  .ngll = ngll,
  .faces = { { .my_orientation = specfem::mesh_entity::dim3::type::right,
               .neighbor_orientation = specfem::mesh_entity::dim3::type::left,
               .theta = 0,
               .my_element = 0,
               .neighbor_element = 1 } }
};

const static auto expected_group_0_2 = ExpectedCommunicationGroup{
  .my_rank = 0,
  .neighbor_rank = 2,
  .nfaces = 4,
  .ngll = ngll,
  .faces = { { .my_orientation = specfem::mesh_entity::dim3::type::back,
               .neighbor_orientation = specfem::mesh_entity::dim3::type::front,
               .theta = 0,
               .my_element = 0,
               .neighbor_element = 2 } }
};

// -- (neighbor_rank=1) --
const static auto expected_group_1_0 = ExpectedCommunicationGroup{
  .my_rank = 1,
  .neighbor_rank = 3,
  .nfaces = 4,
  .ngll = ngll,
  .faces = { { .my_orientation = specfem::mesh_entity::dim3::type::back,
               .neighbor_orientation = specfem::mesh_entity::dim3::type::front,
               .theta = 0,
               .my_element = 1,
               .neighbor_element = 3 } }
};

const static auto expected =
    ExpectedMPICommunicationGroups{ expected_group_0_1, expected_group_0_2,
                                    expected_group_1_0 };

} // namespace TestData::CommunicationGroup::HomogeneousMediumMPI4x4
