#ifndef _UNIT_TESTS_COMPARE_ARRAY_TPP_
#define _UNIT_TESTS_COMPARE_ARRAY_TPP_

#include "IO/fortranio/interface.hpp"
#include "compare_array.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

bool equate_norm(type_real error_norm, type_real computed_norm,
                 type_real tolerance) {
  type_real percent_norm = (error_norm / computed_norm);

  // check nan value
  if (percent_norm != percent_norm) {
    std::cout << "Normalized error is NaN value" << std::endl;

    return false;
  }

  if (percent_norm > tolerance) {
    std::cout << "Normalized error is = " << percent_norm
              << " which is greater than specified tolerance = " << tolerance
              << " computed norm = " << computed_norm
              << " error norm = " << error_norm << std::endl;

    return false;
  }

  return true;
}

} // namespace

template <typename value_type, typename Layout>
specfem::testing::array1d<value_type, Layout>::array1d(
    specfem::kokkos::HostView1d<value_type, Layout> computed_array)
    : n1(computed_array.extent(0)), data(computed_array) {

  value_type max_val = std::numeric_limits<value_type>::min();
  value_type min_val = std::numeric_limits<value_type>::max();

  for (int i1 = 0; i1 < data.extent(0); i1++) {
    max_val = std::max(max_val, data(i1));
    min_val = std::min(min_val, data(i1));
  }

  this->tol = this->tol * fabs(max_val + min_val) / 2;

  return;
}

template <typename value_type, typename Layout>
specfem::testing::array2d<value_type, Layout>::array2d(
    specfem::kokkos::HostView2d<value_type, Layout> computed_array)
    : n1(computed_array.extent(0)), n2(computed_array.extent(1)),
      data(computed_array) {

  value_type max_val = std::numeric_limits<value_type>::min();
  value_type min_val = std::numeric_limits<value_type>::max();

  for (int i1 = 0; i1 < data.extent(0); i1++) {
    for (int i2 = 0; i2 < data.extent(1); i2++) {
      max_val = std::max(max_val, data(i1, i2));
      min_val = std::min(min_val, data(i1, i2));
    }
  }

  this->tol = this->tol * fabs(max_val + min_val) / 2;

  return;
}

template <typename value_type, typename Layout>
specfem::testing::array3d<value_type, Layout>::array3d(
    specfem::kokkos::HostView3d<value_type, Layout> computed_array)
    : n1(computed_array.extent(0)), n2(computed_array.extent(1)),
      n3(computed_array.extent(2)), data(computed_array) {

  value_type max_val = std::numeric_limits<value_type>::min();
  value_type min_val = std::numeric_limits<value_type>::max();

  for (int i1 = 0; i1 < data.extent(0); i1++) {
    for (int i2 = 0; i2 < data.extent(1); i2++) {
      for (int i3 = 0; i3 < data.extent(2); i3++) {
        max_val = std::max(max_val, data(i1, i2, i3));
        min_val = std::min(min_val, data(i1, i2, i3));
      }
    }
  }

  this->tol = this->tol * fabs(max_val + min_val) / 2;

  return;
}

template <typename value_type, typename Layout>
specfem::testing::array1d<value_type, Layout>::array1d(std::string &ref_file,
                                                       const int n1)
    : n1(n1), data("reference", n1) {

  value_type ref_value;
  std::ifstream stream;
  stream.open(ref_file);

  for (int i1 = 0; i1 < n1; i1++) {
    specfem::IO::fortran_read_line(stream, &ref_value);
    data(i1) = ref_value;
  }

  stream.close();

  value_type max_val = std::numeric_limits<value_type>::min();
  value_type min_val = std::numeric_limits<value_type>::max();

  for (int i1 = 0; i1 < data.extent(0); i1++) {
    max_val = std::max(max_val, data(i1));
    min_val = std::min(min_val, data(i1));
  }

  this->tol = this->tol * fabs(max_val + min_val) / 2;

  return;
}

template <typename value_type, typename Layout>
specfem::testing::array2d<value_type, Layout>::array2d(std::string &ref_file,
                                                       const int n1,
                                                       const int n2)
    : n1(n1), n2(n2), data("reference", n1, n2) {

  value_type ref_value;
  std::ifstream stream;
  stream.open(ref_file);

  if constexpr (std::is_same_v<Layout, Kokkos::LayoutRight>) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i2 = 0; i2 < n2; i2++) {
        specfem::IO::fortran_read_line(stream, &ref_value);
        data(i1, i2) = ref_value;
      }
    }
  } else if constexpr (std::is_same_v<Layout, Kokkos::LayoutLeft>) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        specfem::IO::fortran_read_line(stream, &ref_value);
        data(i1, i2) = ref_value;
      }
    }
  }

  stream.close();

  value_type max_val = std::numeric_limits<value_type>::min();
  value_type min_val = std::numeric_limits<value_type>::max();

  for (int i1 = 0; i1 < data.extent(0); i1++) {
    for (int i2 = 0; i2 < data.extent(1); i2++) {
      max_val = std::max(max_val, data(i1, i2));
      min_val = std::min(min_val, data(i1, i2));
    }
  }

  this->tol = this->tol * fabs(max_val + min_val) / 2;

  return;
}

template <typename value_type, typename Layout>
specfem::testing::array3d<value_type, Layout>::array3d(std::string &ref_file,
                                                       const int n1,
                                                       const int n2,
                                                       const int n3)
    : n1(n1), n2(n2), n3(n3), data("reference", n1, n2, n3) {

  value_type ref_value;
  std::ifstream stream;
  stream.open(ref_file);

  if constexpr (std::is_same_v<Layout, Kokkos::LayoutRight>) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i3 = 0; i3 < n3; i3++) {
          specfem::IO::fortran_read_line(stream, &ref_value);
          data(i1, i2, i3) = ref_value;
        }
      }
    }
  } else if constexpr (std::is_same_v<Layout, Kokkos::LayoutLeft>) {
    for (int i3 = 0; i3 < n3; i3++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i1 = 0; i1 < n1; i1++) {
          specfem::IO::fortran_read_line(stream, &ref_value);
          data(i1, i2, i3) = ref_value;
        }
      }
    }
  }

  stream.close();

  value_type max_val = std::numeric_limits<value_type>::min();
  value_type min_val = std::numeric_limits<value_type>::max();

  for (int i1 = 0; i1 < data.extent(0); i1++) {
    for (int i2 = 0; i2 < data.extent(1); i2++) {
      for (int i3 = 0; i3 < data.extent(2); i3++) {
        max_val = std::max(max_val, data(i1, i2, i3));
        min_val = std::min(min_val, data(i1, i2, i3));
      }
    }
  }

  this->tol = this->tol * std::abs(max_val + min_val) / 2;

  return;
}

template <typename value_type, typename Layout>
bool specfem::testing::array1d<value_type, Layout>::equate(
    const value_type &computed, const value_type &reference) {
  if constexpr (std::is_same<value_type, int>::value) {
    return specfem::testing::equate(computed, reference);
  } else if constexpr (std::is_same<value_type, type_real>::value) {
    return specfem::testing::equate(computed, reference, tol);
  }
}

template <typename value_type, typename Layout>
bool specfem::testing::array2d<value_type, Layout>::equate(
    const value_type &computed, const value_type &reference) {
  if constexpr (std::is_same<value_type, int>::value) {
    return specfem::testing::equate(computed, reference);
  } else if constexpr (std::is_same<value_type, type_real>::value) {
    return specfem::testing::equate(computed, reference, tol);
  }
}

template <typename value_type, typename Layout>
bool specfem::testing::array3d<value_type, Layout>::equate(
    const value_type &computed, const value_type &reference) {
  if constexpr (std::is_same<value_type, int>::value) {
    return specfem::testing::equate(computed, reference);
  } else if constexpr (std::is_same<value_type, type_real>::value) {
    return specfem::testing::equate(computed, reference, tol);
  }
}

template <typename value_type, typename Layout1>
template <typename Layout2>
bool specfem::testing::array1d<value_type, Layout1>::operator==(
    const specfem::testing::array1d<value_type, Layout2> &ref) {

  assert(this->data.extent(0) == ref.data.extent(0));

  for (int i1 = 0; i1 < this->data.extent(0); i1++) {
    if (!(equate(this->data(i1), ref.data(i1)))) {
      std::cout << "Mismatch at i1 = " << i1 << " computed = " << this->data(i1)
                << " reference = " << ref.data(i1) << std::endl;
      return false;
    }
  }

  return true;
}

template <typename value_type, typename Layout1>
template <typename Layout2>
bool specfem::testing::array2d<value_type, Layout1>::operator==(
    const array2d<value_type, Layout2> &ref) {

  assert(this->data.extent(0) == ref.data.extent(0));
  assert(this->data.extent(1) == ref.data.extent(1));

  for (int i1 = 0; i1 < this->data.extent(0); i1++) {
    for (int i2 = 0; i2 < this->data.extent(1); i2++) {
      if (!(equate(this->data(i1, i2), ref.data(i1, i2)))) {
        std::cout << "Mismatch at i1 = " << i1 << " i2 = " << i2
                  << " computed = " << this->data(i1, i2)
                  << " reference = " << ref.data(i1, i2) << std::endl;
        return false;
      }
    }
  }

  return true;
}

template <typename value_type, typename Layout1>
template <typename Layout2>
bool specfem::testing::array3d<value_type, Layout1>::operator==(
    const array3d<value_type, Layout2> &ref) {

  assert(this->data.extent(0) == ref.data.extent(0));
  assert(this->data.extent(1) == ref.data.extent(1));
  assert(this->data.extent(2) == ref.data.extent(2));

  for (int i1 = 0; i1 < this->data.extent(0); i1++) {
    for (int i2 = 0; i2 < this->data.extent(1); i2++) {
      for (int i3 = 0; i3 < this->data.extent(2); i3++) {
        if (!(equate(this->data(i1, i2, i3), ref.data(i1, i2, i3)))) {
          std::cout << "Mismatch at i1 = " << i1 << " i2 = " << i2
                    << " i3 = " << i3
                    << " computed = " << this->data(i1, i2, i3)
                    << " reference = " << ref.data(i1, i2, i3) << std::endl;
          return false;
        }
      }
    }
  }

  return true;
}

template <typename value_type, typename Layout>
bool specfem::testing::compare_norm(
    const specfem::testing::array1d<value_type, Layout> &computed_array,
    const specfem::testing::array1d<value_type, Layout> &ref_array,
    const type_real &tolerance) {

  assert(computed_array.data.extent(0) == ref_array.data.extent(0));

  type_real error_norm = 0.0;
  type_real computed_norm = 0.0;

  for (int i1 = 0; i1 < computed_array.data.extent(0); i1++) {
    error_norm += std::abs(computed_array.data(i1) - ref_array.data(i1));
    computed_norm += std::abs(computed_array.data(i1));
  }

  std::cout << "Error norm = " << error_norm
            << " computed norm = " << computed_norm << std::endl;

  return equate_norm(error_norm, computed_norm, tolerance);
}

template <typename value_type, typename Layout>
bool specfem::testing::compare_norm(
    const specfem::testing::array2d<value_type, Layout> &computed_array,
    const specfem::testing::array2d<value_type, Layout> &ref_array,
    const type_real &tolerance) {

  assert(computed_array.data.extent(0) == ref_array.data.extent(0));
  assert(computed_array.data.extent(1) == ref_array.data.extent(1));

  type_real error_norm = 0.0;
  type_real computed_norm = 0.0;

  for (int i1 = 0; i1 < computed_array.data.extent(0); i1++) {
    for (int i2 = 0; i2 < computed_array.data.extent(1); i2++) {
      error_norm +=
          std::abs(computed_array.data(i1, i2) - ref_array.data(i1, i2));
      computed_norm += std::abs(computed_array.data(i1, i2));
    }
  }

  std::cout << "Error norm = " << error_norm
            << " computed norm = " << computed_norm << std::endl;

  return equate_norm(error_norm, computed_norm, tolerance);
}

template <typename value_type, typename Layout>
bool specfem::testing::compare_norm(
    const specfem::testing::array3d<value_type, Layout> &computed_array,
    const specfem::testing::array3d<value_type, Layout> &ref_array,
    const type_real &tolerance) {

  assert(computed_array.data.extent(0) == ref_array.data.extent(0));
  assert(computed_array.data.extent(1) == ref_array.data.extent(1));
  assert(computed_array.data.extent(2) == ref_array.data.extent(2));

  type_real error_norm = 0.0;
  type_real computed_norm = 0.0;

  for (int i1 = 0; i1 < computed_array.data.extent(0); i1++) {
    for (int i2 = 0; i2 < computed_array.data.extent(1); i2++) {
      for (int i3 = 0; i3 < computed_array.data.extent(2); i3++) {
        error_norm += std::abs(computed_array.data(i1, i2, i3) -
                               ref_array.data(i1, i2, i3));
        computed_norm += std::abs(computed_array.data(i1, i2, i3));
      }
    }
  }

  return equate_norm(error_norm, computed_norm, tolerance);
}

#endif
