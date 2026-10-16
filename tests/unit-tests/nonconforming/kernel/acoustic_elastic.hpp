#pragma once
#include "specfem/assembly.hpp"
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace specfem::nonconforming_test::kernel {

void test_nonconforming_acoustic_elastic(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const std::string &meshname);

void test_nonconforming_acoustic_elastic(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname);

} // namespace specfem::nonconforming_test::kernel
