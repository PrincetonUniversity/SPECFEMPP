#pragma once
#include "enumerations/dimension.hpp"
#include "io/fortranio/interface.hpp"
#include "io/mesh/impl/fortran/dim3/generate_database/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace io {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim3 {

// Primary template for try-catch wrapper
/**
 * @brief Try-catch wrapper for fortran_read_line
 *
 * @param message Message to print if exception is caught
 * @param args Arguments to pass to fortran_read_line
 * @throws std::runtime_error if an error occurs while reading the line
 *         includes the input message, so the user know which value errors
 * @see ::specfem::io::fortran_read_line
 * @note This function is a wrapper for fortran_read_line that catches
 * exceptions and throws a runtime_error with a more informative message
 *
 * @code{.cpp}
 * // Example of how to use this function
 * int very_specific_value;
 * try_read_line("very_specific_value", stream, &value);
 * @endcode
 */
template <typename... Args>
auto try_read_line(const std::string &message, Args &&...args)
    -> decltype(specfem::io::fortran_read_line(std::forward<Args>(args)...)) {
  try {
    return specfem::io::fortran_read_line(std::forward<Args>(args)...);
  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message << "Exception caught in try_read_line(" << message
                  << ", ...): " << e.what();
    throw std::runtime_error(error_message.str());
  } catch (...) {
    std::ostringstream error_message;
    error_message << "Unknown exception caught in " << message;
    throw std::runtime_error(error_message.str());
  }
}

/**
 * @brief Try-catch wrapper for read_array
 *
 * @param message Message to print if exception is caught
 * @param args Arguments to pass to read_array
 * @throws std::runtime_error if an error occurs while reading the array
 *         includes the input message, so the user know which array errors
 * @see ::specfem::io::mesh::impl::fortran::dim3::read_array
 * @note This function is a wrapper for read_array that catches exceptions
 *       and throws a runtime_error with a more informative message
 *
 * @code{.cpp}
 * // Example of how to use this function
 * Kokkos::View<int *, Kokkos::HostSpace> specific_array("array", 10);
 * try_read_array("specific_array", stream, array);
 * @endcode
 */
template <typename... Args>
auto try_read_array(const std::string &message, Args &&...args)
    -> decltype(specfem::io::mesh::impl::fortran::dim3::read_array(
        std::forward<Args>(args)...)) {
  try {
    return specfem::io::mesh::impl::fortran::dim3::read_array(
        std::forward<Args>(args)...);
  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message << "Exception caught in try_read_array('" << message
                  << "', ...): " << e.what();
    throw std::runtime_error(error_message.str());
  } catch (...) {
    std::ostringstream error_message;
    error_message << "Unknown exception caught in try_read_array('" << message
                  << "', ...).";
    throw std::runtime_error(error_message.str());
  }
}

/**
 * @brief Try-catch wrapper for read_index_array
 *
 * @param message Message to print if exception is caught
 * @param args Arguments to pass to read_index_array
 * @throws std::runtime_error if an error occurs while reading the array that
 *         includes the input message, so the user know which array errors
 * @see ::specfem::io::mesh::impl::fortran::dim3::read_index_array
 * @note This function is a wrapper for read_index_array that catches exceptions
 *       and throws a runtime_error with a more informative message
 *
 * @code{.cpp}
 * // Example of how to use this function
 * Kokkos::View<int *, Kokkos::HostSpace> specific_array("array", 10);
 * try_read_index_array("specific_array", stream, array);
 * @endcode
 */
template <typename... Args>
auto try_read_index_array(const std::string &message, Args &&...args)
    -> decltype(specfem::io::mesh::impl::fortran::dim3::read_index_array(
        std::forward<Args>(args)...)) {
  try {
    return specfem::io::mesh::impl::fortran::dim3::read_index_array(
        std::forward<Args>(args)...);
  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message << "Exception caught in try_read_index('" << message
                  << "', ...): " << e.what();
    throw std::runtime_error(error_message.str());
  } catch (...) {
    std::ostringstream error_message;
    error_message << "Unknown exception caught in try_read_index('" << message
                  << "', ...).";
    throw std::runtime_error(error_message.str());
  }
}

template <typename... Args>
auto try_read_control_nodes_indexing(const std::string &message, Args &&...args)
    -> decltype(
        specfem::io::mesh::impl::fortran::dim3::read_control_nodes_indexing(
            std::forward<Args>(args)...)) {
  try {
    return specfem::io::mesh::impl::fortran::dim3::read_control_nodes_indexing(
        std::forward<Args>(args)...);
  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message << "Exception caught in try_read_control_nodes_indexing('"
                  << message << "', ...): " << e.what();
    throw std::runtime_error(error_message.str());
  } catch (...) {
    std::ostringstream error_message;
    error_message
        << "Unknown exception caught in try_read_control_nodes_indexing('"
        << message << "', ...).";
    throw std::runtime_error(error_message.str());
  }
}

template <typename... Args>
auto try_read_control_nodes_coordinates(const std::string &message,
                                        Args &&...args)
    -> decltype(
        specfem::io::mesh::impl::fortran::dim3::read_control_nodes_coordinates(
            std::forward<Args>(args)...)) {
  try {
    return specfem::io::mesh::impl::fortran::dim3::
        read_control_nodes_coordinates(std::forward<Args>(args)...);
  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message << "Exception caught in try_read_control_nodes_coordinates('"
                  << message << "', ...): " << e.what();
    throw std::runtime_error(error_message.str());
  } catch (...) {
    std::ostringstream error_message;
    error_message
        << "Unknown exception caught in try_read_control_nodes_coordinates('"
        << message << "', ...).";
    throw std::runtime_error(error_message.str());
  }
}

} // namespace dim3
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace io
} // namespace specfem
