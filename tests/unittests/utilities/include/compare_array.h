/***
 * Routines to facilitate testing infrastructure
 ***/

#ifndef COMPARE_ARRAY_H
#define COMPARE_ARRAY_H

#include "../../../../include/kokkos_abstractions.h"
#include <string>

namespace specfem {
namespace testing {

void equate(int computed_value, int ref_value);

void equate(type_real computed_value, type_real ref_value, type_real tol);

void test_array(specfem::HostView1d<int> computed_array, std::string ref_file,
                int n1);

void test_array(specfem::HostView2d<int> computed_array, std::string ref_file,
                int n1, int n2);

void test_array(specfem::HostView3d<int> computed_array, std::string ref_file,
                int n1, int n2, int n3);

void test_array(specfem::HostView1d<type_real> computed_array,
                std::string ref_file, int n1);

void test_array(specfem::HostView2d<type_real> computed_array,
                std::string ref_file, int n1, int n2);

void test_array(specfem::HostView3d<type_real> computed_array,
                std::string ref_file, int n1, int n2, int n3);
} // namespace testing
} // namespace specfem

#endif
