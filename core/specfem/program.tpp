#pragma once

#include "specfem/element.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/version.hpp"
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

  auto center = [](const std::string &text, std::size_t width) {
    if (text.length() >= width) {
      return text;
    }
    const std::size_t padding = (width - text.length()) / 2;
    return std::string(padding, ' ') + text;
  };

  message << specfem::version::header() << "\n"
          << center(dim + " SIMULATION", 58) << "\n"
          << "\n"
          << std::string(58, '-') << "\n"
          << "Title : " << setup.get_header().get_title() << "\n"
          << "Description: " << setup.get_header().get_description() << "\n"
          << "Simulation start time: " << ctime(&c_now)
          << std::string(58, '-') << "\n\n";

  return message.str();

}

} // namespace specfem::program
