#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/compute/compute_derivatives.tpp"
#include "specfem/io.hpp"
#include <gtest/gtest.h>

namespace compute_derivatives_test_impl {

template <typename Tags> class ComputeDerivativesTest : public ::testing::Test {
protected:
  static constexpr auto dimension = Tags::dimension_tag;
  static constexpr auto medium = Tags::medium_tag;
  static constexpr auto property = Tags::property_tag;
  static constexpr int ngll = 5;
  // An odd element count also exercises partial SIMD chunks on CPUs.
  static constexpr int nspec = 3;
  static constexpr type_real dt = 0.1;

  using PointTags = specfem::tags::Tags<dimension, medium, property, false>;
  using Properties = specfem::point::properties<PointTags>;
  using Kernels = specfem::point::kernels<PointTags>;
  using Index = specfem::point::index<dimension>;

  static specfem::assembly::assembly<dimension> make_assembly();
  specfem::assembly::assembly<dimension> assembly = make_assembly();

  void SetUp() override;
  static Properties make_properties();
  static type_real expected_kernel();
  static type_real selected_kernel(const Kernels &kernels);

  template <typename Functor> void for_each_point(Functor &&functor) const;
  void set_fields(type_real acceleration, type_real displacement);
  void compute();
};

} // namespace compute_derivatives_test_impl

#include "test_fixture.tpp"
