#include "source.hpp"
#include <gtest/gtest.h>

// Test all source types with their parameters and solutions
TEST(SourceTests, Sources2D) {
  // 2D Force Source
  {
    auto parameters_and_solutions = get_parameters_and_solutions<
        specfem::dimension::type::dim2,
        specfem::sources::force<specfem::dimension::type::dim2> >();
    for (const auto &param_solution : parameters_and_solutions) {
      const auto &parameters = std::get<0>(param_solution);
      const auto &solution = std::get<1>(param_solution);
      test_source<specfem::dimension::type::dim2,
                  specfem::sources::force<specfem::dimension::type::dim2> >(
          parameters, solution);
    }
  }

  // 2D External Source
  {
    auto parameters_and_solutions = get_parameters_and_solutions<
        specfem::dimension::type::dim2,
        specfem::sources::external<specfem::dimension::type::dim2> >();
    for (const auto &param_solution : parameters_and_solutions) {
      const auto &parameters = std::get<0>(param_solution);
      const auto &solution = std::get<1>(param_solution);
      test_source<specfem::dimension::type::dim2,
                  specfem::sources::external<specfem::dimension::type::dim2> >(
          parameters, solution);
    }
  }

  // 2D Cosserat Force Source
  {
    auto parameters_and_solutions = get_parameters_and_solutions<
        specfem::dimension::type::dim2,
        specfem::sources::cosserat_force<specfem::dimension::type::dim2> >();
    for (const auto &param_solution : parameters_and_solutions) {
      const auto &parameters = std::get<0>(param_solution);
      const auto &solution = std::get<1>(param_solution);
      test_source<
          specfem::dimension::type::dim2,
          specfem::sources::cosserat_force<specfem::dimension::type::dim2> >(
          parameters, solution);
    }
  }

  // 2D Adjoint Source
  {
    auto parameters_and_solutions = get_parameters_and_solutions<
        specfem::dimension::type::dim2,
        specfem::sources::adjoint_source<specfem::dimension::type::dim2> >();
    for (const auto &param_solution : parameters_and_solutions) {
      const auto &parameters = std::get<0>(param_solution);
      const auto &solution = std::get<1>(param_solution);
      test_source<
          specfem::dimension::type::dim2,
          specfem::sources::adjoint_source<specfem::dimension::type::dim2> >(
          parameters, solution);
    }
  }

  // 2D Moment Tensor Source
  {
    auto parameters_and_solutions = get_parameters_and_solutions<
        specfem::dimension::type::dim2,
        specfem::sources::moment_tensor<specfem::dimension::type::dim2> >();
    for (const auto &param_solution : parameters_and_solutions) {
      const auto &parameters = std::get<0>(param_solution);
      const auto &solution = std::get<1>(param_solution);
      test_source<
          specfem::dimension::type::dim2,
          specfem::sources::moment_tensor<specfem::dimension::type::dim2> >(
          parameters, solution);
    }
  }
}

TEST(SourceTests, Sources3D) {
  // 3D Force Source
  {
    auto parameters_and_solutions = get_parameters_and_solutions<
        specfem::dimension::type::dim3,
        specfem::sources::force<specfem::dimension::type::dim3> >();
    for (const auto &param_solution : parameters_and_solutions) {
      const auto &parameters = std::get<0>(param_solution);
      const auto &solution = std::get<1>(param_solution);
      test_source<specfem::dimension::type::dim3,
                  specfem::sources::force<specfem::dimension::type::dim3> >(
          parameters, solution);
    }
  }

  // 3D Moment Tensor Source
  {
    auto parameters_and_solutions = get_parameters_and_solutions<
        specfem::dimension::type::dim3,
        specfem::sources::moment_tensor<specfem::dimension::type::dim3> >();
    for (const auto &param_solution : parameters_and_solutions) {
      const auto &parameters = std::get<0>(param_solution);
      const auto &solution = std::get<1>(param_solution);
      test_source<
          specfem::dimension::type::dim3,
          specfem::sources::moment_tensor<specfem::dimension::type::dim3> >(
          parameters, solution);
    }
  }
}
