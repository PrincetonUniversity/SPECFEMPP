#pragma once
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

template <specfem::dimension::type DimensionTag, typename SourceType>
struct source_parameters;

template <specfem::dimension::type DimensionTag, typename SourceType>
struct source_solution;

template <specfem::dimension::type DimensionTag, typename SourceType>
std::vector<std::tuple<source_parameters<DimensionTag, SourceType>,
                       source_solution<DimensionTag, SourceType> > >
get_parameters_and_solutions();

// SFINAE helper to detect if SourceType has get_force_vector method (vector
// sources)
template <typename T> struct has_get_force_vector {
private:
  template <typename U>
  static auto test(int)
      -> decltype(std::declval<U>().get_force_vector(), std::true_type{});
  template <typename> static auto test(...) -> std::false_type;

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

// SFINAE helper to detect if SourceType has get_source_tensor method (tensor
// sources)
template <typename T> struct has_get_source_tensor {
private:
  template <typename U>
  static auto test(int)
      -> decltype(std::declval<U>().get_source_tensor(), std::true_type{});
  template <typename> static auto test(...) -> std::false_type;

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

// Factory function template declaration - needs to be specialized for each
// source type
template <specfem::dimension::type DimensionTag, typename SourceType>
SourceType
create_source(const source_parameters<DimensionTag, SourceType> &parameters);

// Generic test function implementation
template <specfem::dimension::type DimensionTag, typename SourceType>
void test_source(const source_parameters<DimensionTag, SourceType> &parameters,
                 const source_solution<DimensionTag, SourceType> &solution) {
  // Add to trace for easier debugging
  std::cout << "Testing source: " << parameters.name << "\n"
            << "  Coordinates: \n"
            << "    x = " << type_real(parameters.x) << "\n";
  if constexpr (DimensionTag == specfem::dimension::type::dim3) {
    std::cout << "    y = " << type_real(parameters.y) << "\n";
  }
  std::cout << "    z = " << type_real(parameters.z) << "\n"
            << "  Medium Tag: "
            << specfem::element::to_string(parameters.medium_tag) << "\n";

  // Create the source - this needs to be specialized per source type
  // For now, we'll use a factory function approach that should be specialized
  auto source = create_source<DimensionTag, SourceType>(parameters);

  // Set the medium tag
  source.set_medium_tag(parameters.medium_tag);

  // Test common coordinate access methods
  EXPECT_REAL_EQ(parameters.x, source.get_x());
  EXPECT_REAL_EQ(parameters.z, source.get_z());
  if constexpr (DimensionTag == specfem::dimension::type::dim3) {
    EXPECT_REAL_EQ(parameters.y, source.get_y());
  }

  // Test source-specific functionality using SFINAE
  if constexpr (has_get_force_vector<decltype(source)>::value) {

    // Vector source testing
    auto computed_force_vector = source.get_force_vector();
    auto expected_force_vector = solution.force_vector;

    EXPECT_EQ(computed_force_vector.extent(0), expected_force_vector.extent(0));

    for (size_t i = 0; i < expected_force_vector.size(); ++i) {
      EXPECT_NEAR(computed_force_vector(i), expected_force_vector(i), 1e-5)
          << "Mismatch at index " << i;
    }
  } else if constexpr (has_get_source_tensor<decltype(source)>::value) {
    // Tensor source testing
    auto computed_source_tensor = source.get_source_tensor();
    auto expected_source_tensor = solution.source_tensor;

    EXPECT_EQ(computed_source_tensor.extent(0),
              expected_source_tensor.extent(0));
    EXPECT_EQ(computed_source_tensor.extent(1),
              expected_source_tensor.extent(1));

    for (size_t i = 0; i < expected_source_tensor.extent(0); ++i) {
      for (size_t j = 0; j < expected_source_tensor.extent(1); ++j) {
        EXPECT_NEAR(computed_source_tensor(i, j), expected_source_tensor(i, j),
                    1e-5)
            << "Mismatch at index (" << i << ", " << j << ")";
      }
    }
  }
}
