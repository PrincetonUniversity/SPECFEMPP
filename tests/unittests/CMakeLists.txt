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
  Kokkos_Environment.hpp
  Kokkos_Environment.cpp
)

target_link_libraries(
  kokkos_environment
  Kokkos::kokkos
  gtest_main
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
)

include(GoogleTest)
gtest_discover_tests(gll_tests)
gtest_discover_tests(lagrange_tests)
gtest_discover_tests(fortranio_test)
