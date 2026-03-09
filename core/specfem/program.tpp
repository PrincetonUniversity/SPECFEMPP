#pragma once

#include "specfem/element.hpp"
#include "specfem/runtime_configuration.hpp"
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>

namespace specfem::program {

template <specfem::element::dimension_tag DimensionTag>
std::string
print_header(const specfem::runtime_configuration::setup &setup,
             const std::chrono::time_point<std::chrono::system_clock> now) {

  std::ostringstream message;

  // convert now to string form
  const std::time_t c_now = std::chrono::system_clock::to_time_t(now);

  std::string dim;

  if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
    dim = "2D";
  } else if constexpr (DimensionTag == specfem::element::dimension_tag::dim3) {
    dim = "3D";
  } else {
    throw std::runtime_error("Unsupported dimension for header print.");
  }

  message << "================================================\n"
          << "            SPECFEM++ " << dim << " SIMULATION\n"
          << "================================================\n\n"
          << "Title : " << setup.get_header().get_title() << "\n"
          << "Discription: " << setup.get_header().get_description() << "\n"
          << "Simulation start time: " << ctime(&c_now)
          << "------------------------------------------------\n";

  return message.str();
}

} // namespace specfem::program
