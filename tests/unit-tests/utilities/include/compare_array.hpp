/***
 * Routines to facilitate testing infrastructure
 ***/

#ifndef COMPARE_ARRAY_H
#define COMPARE_ARRAY_H

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <string>

namespace specfem {
namespace testing {

bool equate(int computed_value, int ref_value);

bool equate(type_real computed_value, type_real ref_value, type_real tol);

template <typename value_type, typename Layout> struct array1d {
  specfem::kokkos::HostView1d<value_type, Layout> data;
  int n1;

  type_real tol = 1e-2;

  bool equate(const value_type &computed_value, const value_type &ref_value);

  array1d(const specfem::kokkos::HostView1d<value_type, Layout> computed_array);

  array1d(std::string &ref_file, const int n1);

  array1d(const int n1) : n1(n1), data("specfem::testing::array1d", n1) {}

  bool
  operator==(const specfem::testing::array1d<value_type, Layout> &ref_array);
};

template <typename value_type, typename Layout> struct array2d {
  specfem::kokkos::HostView2d<value_type, Layout> data;
  int n1, n2;

  type_real tol = 1e-2;

  bool equate(const value_type &computed_value, const value_type &ref_value);

  array2d(const specfem::kokkos::HostView2d<value_type, Layout> computed_array);

  array2d(std::string &ref_file, const int n1, const int n2);

  array2d(const int n1, const int n2)
      : n1(n1), n2(n2), data("specfem::testing::array2d", n1, n2) {}

  bool
  operator==(const specfem::testing::array2d<value_type, Layout> &ref_array);
};

template <typename value_type, typename Layout> struct array3d {
  specfem::kokkos::HostView3d<value_type, Layout> data;
  int n1, n2, n3;

  type_real tol = 1e-2;

  bool equate(const value_type &computed_value, const value_type &ref_value);

  array3d(const specfem::kokkos::HostView3d<value_type, Layout> computed_array);

  array3d(std::string &ref_file, const int n1, const int n2, const int n3);

  array3d(const int n1, const int n2, const int n3)
      : n1(n1), n2(n2), n3(n3), data("specfem::testing::array3d", n1, n2, n3) {}

  bool
  operator==(const specfem::testing::array3d<value_type, Layout> &ref_array);
};

template <typename value_type, typename Layout>
bool compare_norm(
    const specfem::testing::array1d<value_type, Layout> &compute_array,
    const specfem::testing::array1d<value_type, Layout> &ref_array,
    const type_real &tol);

template <typename value_type, typename Layout>
bool compare_norm(
    const specfem::testing::array2d<value_type, Layout> &compute_array,
    const specfem::testing::array2d<value_type, Layout> &ref_array,
    const type_real &tol);

template <typename value_type, typename Layout>
bool compare_norm(
    const specfem::testing::array3d<value_type, Layout> &compute_array,
    const specfem::testing::array3d<value_type, Layout> &ref_array,
    const type_real &tol);

// template <typename T, typename Layout>
// void test_array(specfem::kokkos::HostView1d<T, Layout> computed_array,
//                 std::string ref_file, int n1);

// template <typename T, typename Layout>
// void test_array(specfem::kokkos::HostView2d<T, Layout> computed_array,
//                 std::string ref_file, int n1, int n2);

// template <typename T, typename Layout>
// void test_array(specfem::kokkos::HostView3d<T, Layout> computed_array,
//                 std::string ref_file, int n1, int n2, int n3);

// template <typename T, typename Layout>
// void compare_norm(specfem::kokkos::HostView1d<T, Layout> computed_array,
//                   std::string ref_file, int n1, T tolerance);

// template <typename T, typename Layout>
// void compare_norm(specfem::kokkos::HostView2d<T, Layout> computed_array,
//                   std::string ref_file, int n1, int n2, T tolerance);

// template <typename T, typename Layout>
// void compare_norm(specfem::kokkos::HostView3d<T, Layout> computed_array,
//                   std::string ref_file, int n1, int n2, int n3, T tolerance);

// void test_array(
//     specfem::kokkos::HostView1d<int, Kokkos::LayoutRight> computed_array,
//     std::string ref_file, int n1);

// void test_array(
//     specfem::kokkos::HostView2d<int, Kokkos::LayoutRight> computed_array,
//     std::string ref_file, int n1, int n2);

// void test_array(
//     specfem::kokkos::HostView3d<int, Kokkos::LayoutRight> computed_array,
//     std::string ref_file, int n1, int n2, int n3);

// void test_array(
//     specfem::kokkos::HostView1d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1);

// void test_array(
//     specfem::kokkos::HostView2d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, int n2);

// void test_array(
//     specfem::kokkos::HostView3d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, int n2, int n3);

// void compare_norm(
//     specfem::kokkos::HostView1d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, type_real tolerance);

// void compare_norm(
//     specfem::kokkos::HostView2d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, int n2, type_real
//     tolerance);

// void compare_norm(
//     specfem::kokkos::HostView3d<type_real, Kokkos::LayoutRight>
//     computed_array, std::string ref_file, int n1, int n2, int n3, type_real
//     tolerance);

// void test_array(
//     specfem::kokkos::HostView1d<int, Kokkos::LayoutLeft> computed_array,
//     std::string ref_file, int n1);

// void test_array(
//     specfem::kokkos::HostView2d<int, Kokkos::LayoutLeft> computed_array,
//     std::string ref_file, int n1, int n2);

// void test_array(
//     specfem::kokkos::HostView3d<int, Kokkos::LayoutLeft> computed_array,
//     std::string ref_file, int n1, int n2, int n3);

// void test_array(
//     specfem::kokkos::HostView1d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1);

// void test_array(
//     specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, int n2);

// void test_array(
//     specfem::kokkos::HostView3d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, int n2, int n3);

// void compare_norm(
//     specfem::kokkos::HostView1d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, type_real tolerance);

// void compare_norm(
//     specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, int n2, type_real
//     tolerance);

// void compare_norm(
//     specfem::kokkos::HostView3d<type_real, Kokkos::LayoutLeft>
//     computed_array, std::string ref_file, int n1, int n2, int n3, type_real
//     tolerance);
} // namespace testing
} // namespace specfem

#endif
