#include "io/mesh/impl/fortran/dim3/generate_database/interface.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Rewrite check_read_test_value to use try_read_line
void specfem::io::mesh::impl::fortran::dim3::check_read_test_value(
    std::ifstream &stream, int test_value) {
  // Read test value that should be value
  int value;
  try_read_line("check_read_test_value", stream, &value);
  if (test_value != value) {
    std::ostringstream error_message;
    error_message << "Test value (" << test_value << ") != read value ("
                  << value << "). "
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}

void specfem::io::mesh::impl::fortran::dim3::check_values(std::string message,
                                                          int value,
                                                          int expected) {
  if (value != expected) {
    std::ostringstream error_message;
    error_message << message << " value (" << value << ") != expected value ("
                  << expected << "). "
                  << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}
