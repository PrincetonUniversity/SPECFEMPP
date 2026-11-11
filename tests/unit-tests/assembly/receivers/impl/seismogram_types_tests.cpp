#include "specfem/assembly/receivers/impl/seismogram_types.hpp"
#include "specfem/enums.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace seismogram_types_test_impl {

std::vector<specfem::enums::wavefield> configured_types() {
  return { specfem::enums::wavefield::velocity,
           specfem::enums::wavefield::pressure,
           specfem::enums::wavefield::displacement };
}

} // namespace seismogram_types_test_impl

TEST(SeismogramTypes, ValidTypesIndexInOrder) {
  const specfem::assembly::receivers_impl::SeismogramTypes seismogram_types(
      seismogram_types_test_impl::configured_types());
  EXPECT_EQ(seismogram_types.size(), 3);
  EXPECT_EQ(seismogram_types.index_of(specfem::enums::wavefield::velocity), 0);
  EXPECT_EQ(seismogram_types.index_of(specfem::enums::wavefield::pressure), 1);
  EXPECT_EQ(seismogram_types.index_of(specfem::enums::wavefield::displacement),
            2);
  EXPECT_EQ(seismogram_types.types(),
            seismogram_types_test_impl::configured_types());
}

TEST(SeismogramTypes, UnknownTypeThrows) {
  const specfem::assembly::receivers_impl::SeismogramTypes seismogram_types(
      seismogram_types_test_impl::configured_types());
  EXPECT_THROW(
      seismogram_types.index_of(specfem::enums::wavefield::acceleration),
      std::runtime_error);
}

TEST(SeismogramTypes, TypeAtRoundTrips) {
  const specfem::assembly::receivers_impl::SeismogramTypes seismogram_types(
      seismogram_types_test_impl::configured_types());
  for (int index = 0; index < seismogram_types.size(); ++index) {
    EXPECT_EQ(seismogram_types.index_of(seismogram_types.type_at(index)),
              index);
  }
}
