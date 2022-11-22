cmake_minimum_required(VERSION 3.14)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_library(
  kokkos_environment
  Kokkos_Environment.cpp
)

target_link_libraries(
  kokkos_environment
  Kokkos::kokkos
  gtest_main
)

add_library(
  mpi_environment
  MPI_environment.cpp
)

target_link_libraries(
  mpi_environment
  gtest_main
  specfem_mpi
)

add_library(
  compare_arrays
  utilities/src/compare_array.cpp
)

target_link_libraries(
  compare_arrays
  Kokkos::kokkos
  fortranio
)


add_executable(
  gll_tests
  gll/gll_tests.cpp
)
target_link_libraries(
  gll_tests
  gtest_main
  gll_library
  gll_utils
  kokkos_environment
  -lpthread -lm
)

add_executable(
  lagrange_tests
  lagrange/Lagrange_tests.cpp
)

target_link_libraries(
  lagrange_tests
  gtest_main
  lagrange
  gll_library
  kokkos_environment
  -lpthread -lm
)

add_executable(
  fortranio_test
  fortran_io/fortranio_tests.cpp
)

target_link_libraries(
  fortranio_test
  gtest_main
  gmock_main
  fortranio
  -lpthread -lm
)

add_executable(
  mesh_tests
  mesh/mesh_tests.cpp
)

target_link_libraries(
  mesh_tests
  gtest_main
  mesh
  mpi_environment
  kokkos_environment
  yaml-cpp
  material_class
  -lpthread -lm
)

add_executable(
  compute_partial_derivatives_tests
  compute/compute_partial_derivatives_tests.cpp
)

target_link_libraries(
  compute_partial_derivatives_tests
  mesh
  compute
  quadrature
  mpi_environment
  kokkos_environment
  yaml-cpp
  compare_arrays
  material_class
  -lpthread -lm
)

add_executable(
  compute_properties_tests
  compute/compute_properties_tests.cpp
)

target_link_libraries(
  compute_properties_tests
  mesh
  compute
  quadrature
  mpi_environment
  kokkos_environment
  yaml-cpp
  compare_arrays
  material_class
  -lpthread -lm
)

add_executable(
  compute_tests
  compute/compute_tests.cpp
)

target_link_libraries(
  compute_tests
  mesh
  compute
  quadrature
  mpi_environment
  kokkos_environment
  yaml-cpp
  compare_arrays
  material_class
  -lpthread -lm
)


include(GoogleTest)
gtest_discover_tests(gll_tests)
gtest_discover_tests(lagrange_tests)
gtest_discover_tests(fortranio_test)
gtest_discover_tests(mesh_tests)
gtest_discover_tests(compute_partial_derivatives_tests)
gtest_discover_tests(compute_properties_tests)
gtest_discover_tests(compute_tests)
