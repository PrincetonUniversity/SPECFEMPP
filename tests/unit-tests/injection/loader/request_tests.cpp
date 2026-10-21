// Tests for specfem::injection::Request: the slot layout of the incident-field
// buffer.

#include "specfem/injection.hpp"
#include <gtest/gtest.h>

#include <array>
#include <span>
#include <stdexcept>

namespace specfem::injection_loader_test {

static_assert(specfem::injection::component_count(spfp_quantity_velocity) == 3);
static_assert(
    specfem::injection::component_count(spfp_quantity_potential_rate) == 1);

constexpr std::array<spfp_quantity, 2> elastic_request = {
  spfp_quantity_velocity, spfp_quantity_traction
};
static_assert(specfem::injection::count_components(elastic_request) == 6);
static_assert(specfem::injection::count_components(
                  std::span(elastic_request).first(1)) == 3);

TEST(InjectionRequest, ComponentCounts) {
  EXPECT_EQ(specfem::injection::component_count(spfp_quantity_velocity), 3);
  EXPECT_EQ(specfem::injection::component_count(spfp_quantity_traction), 3);
  EXPECT_EQ(specfem::injection::component_count(spfp_quantity_potential_rate),
            1);
  EXPECT_THROW([[maybe_unused]] const auto count =
                   specfem::injection::component_count(99),
               std::runtime_error);
}

TEST(InjectionRequest, SlotLayout) {
  const specfem::injection::Request request{
    { spfp_quantity_velocity, spfp_quantity_traction },
    { spfp_quantity_potential_rate, spfp_quantity_displacement }
  };
  EXPECT_EQ(request.components(spfp_medium_elastic), 6);
  EXPECT_EQ(request.components(spfp_medium_acoustic), 4);
  EXPECT_EQ(request.max_components(), 6);

  EXPECT_EQ(request.slot(spfp_medium_elastic, 0), 0);
  EXPECT_EQ(request.slot(spfp_medium_elastic, 1), 3);
  EXPECT_EQ(request.slot(spfp_medium_acoustic, 1), 1);
  EXPECT_THROW([[maybe_unused]] const auto slot =
                   request.slot(spfp_medium_elastic, 2),
               std::out_of_range);
  EXPECT_THROW([[maybe_unused]] const auto &list = request.quantities(7),
               std::runtime_error);
}

TEST(InjectionRequest, EmptyRequest) {
  const specfem::injection::Request request;
  EXPECT_EQ(request.max_components(), 0);
  EXPECT_EQ(request.components(spfp_medium_acoustic), 0);
}

} // namespace specfem::injection_loader_test
