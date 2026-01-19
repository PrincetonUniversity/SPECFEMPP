#include "../test_fixture.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/shape_function.hpp"
#include "specfem/utilities.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace specfem::assembly_test {

struct ShapeFunction3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  int ngllz;              ///< Number of GLL points in z-direction
  int nglly;              ///< Number of GLL points in y-direction
  int ngllx;              ///< Number of GLL points in x-direction
  int nnodes_per_element; ///< Number of control nodes per element

  ShapeFunction3D(int ngllz, int nglly, int ngllx, int nnodes_per_element)
      : ngllz(ngllz), nglly(nglly), ngllx(ngllx),
        nnodes_per_element(nnodes_per_element) {}

  void check(const specfem::assembly::mesh_impl::shape_functions<dimension>
                 &shape_functions) const {

    // Gernerate quadrature
    const auto quadrature = []() {
      specfem::quadrature::gll::gll gll{};
      return specfem::quadrature::quadratures(gll);
    }();

    ASSERT_EQ(shape_functions.ngllz, ngllz)
        << "Number of GLL points in z-direction mismatch. "
        << "Expected: " << ngllz << ", "
        << "Got: " << shape_functions.ngllz << std::endl;
    ASSERT_EQ(shape_functions.nglly, nglly)
        << "Number of GLL points in y-direction mismatch. "
        << "Expected: " << nglly << ", "
        << "Got: " << shape_functions.nglly << std::endl;
    ASSERT_EQ(shape_functions.ngllx, ngllx)
        << "Number of GLL points in x-direction mismatch. "
        << "Expected: " << ngllx << ", "
        << "Got: " << shape_functions.ngllx << std::endl;
    ASSERT_EQ(shape_functions.ngnod, nnodes_per_element)
        << "Number of control nodes per element mismatch. "
        << "Expected: " << nnodes_per_element << ", "
        << "Got: " << shape_functions.ngnod << std::endl;

    // Check if the shape function views are allocated correctly
    ASSERT_TRUE(shape_functions.shape3D.extent(0) == ngllz)
        << "Shape function extent 0 mismatch.";
    ASSERT_TRUE(shape_functions.shape3D.extent(1) == nglly)
        << "Shape function extent 1 mismatch.";
    ASSERT_TRUE(shape_functions.shape3D.extent(2) == ngllx)
        << "Shape function extent 2 mismatch.";
    ASSERT_TRUE(shape_functions.shape3D.extent(3) == nnodes_per_element)
        << "Shape function extent 3 mismatch.";
    ASSERT_TRUE(shape_functions.dshape3D.extent(0) == ngllz)
        << "Shape function derivative extent 0 mismatch.";
    ASSERT_TRUE(shape_functions.dshape3D.extent(1) == nglly)
        << "Shape function derivative extent 1 mismatch.";
    ASSERT_TRUE(shape_functions.dshape3D.extent(2) == ngllx)
        << "Shape function derivative extent 2 mismatch.";
    ASSERT_TRUE(shape_functions.dshape3D.extent(3) == 3)
        << "Shape function derivative extent 3 mismatch.";
    ASSERT_TRUE(shape_functions.dshape3D.extent(4) == nnodes_per_element)
        << "Shape function derivative extent 4 mismatch.";

    const auto xi = quadrature.gll.get_hxi();
    for (int iz = 0; iz < ngllz; iz++) {
      for (int iy = 0; iy < nglly; iy++) {
        for (int ix = 0; ix < ngllx; ix++) {
          type_real xil = xi(ix);
          type_real etal = xi(iy);
          type_real zetal = xi(iz);

          const auto expected_shape_function =
              specfem::shape_function::shape_function(xil, etal, zetal,
                                                      nnodes_per_element);

          for (int in = 0; in < nnodes_per_element; in++) {
            if (!specfem::utilities::is_close(
                    shape_functions.h_shape3D(iz, iy, ix, in),
                    expected_shape_function[in])) {
              FAIL() << "Shape function value mismatch at (" << iz << ", " << iy
                     << ", " << ix << ") for node " << in << ". "
                     << "Expected: " << expected_shape_function[in] << ", "
                     << "Got: " << shape_functions.h_shape3D(iz, iy, ix, in)
                     << std::endl;
            }
            return;
          }

          const auto expected_shape_function_derivatives =
              specfem::shape_function::shape_function_derivatives(
                  xil, etal, zetal, nnodes_per_element);

          for (int in = 0; in < nnodes_per_element; in++) {
            for (int dim = 0; dim < 3; dim++) {
              if (!specfem::utilities::is_close(
                      shape_functions.h_dshape3D(iz, iy, ix, dim, in),
                      expected_shape_function_derivatives[dim][in])) {
                FAIL() << "Shape function derivative mismatch at (" << iz
                       << ", " << iy << ", " << ix << ") for node " << in
                       << " in dimension " << dim << ". "
                       << "Expected: "
                       << expected_shape_function_derivatives[dim][in] << ", "
                       << "Got: "
                       << shape_functions.h_dshape3D(iz, iy, ix, dim, in)
                       << std::endl;
              }
            }
          }
        }
      }
    }
  }
};
} // namespace specfem::assembly_test

using namespace specfem::assembly_test;

const static std::unordered_map<std::string,
                                specfem::assembly_test::ShapeFunction3D>
    expected_shape_functions_map = { { "EightNodeElastic",
                                       specfem::assembly_test::ShapeFunction3D(
                                           5, 5, 5, 8) } };

TEST_P(Assembly3DTest, ShapeFunctions) {
  const auto &param_name = GetParam();
  if (expected_shape_functions_map.find(param_name) ==
      expected_shape_functions_map.end()) {
    GTEST_SKIP() << "No expected shape function data found for parameter: "
                 << param_name;
    return;
  }

  const auto &expected_shape_functions =
      expected_shape_functions_map.at(param_name);
  const auto &assembly = getAssembly();
  const auto &shape_functions = assembly.mesh;

  expected_shape_functions.check(shape_functions);
}
