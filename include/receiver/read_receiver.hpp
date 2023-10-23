#ifndef _READ_RECEIVER_HPP
#define _READ_RECEIVER_HPP

#include "receiver.hpp"
#include "specfem_setup.hpp"
#include <vector>

namespace specfem {
namespace receivers {
/**
 * @brief Read receiver station file
 *
 * Parse receiver stations file and create a vector of
 * specfem::source::source * object
 *
 * @param stations_file Stations file describing receiver locations
 * @param angle Angle of the receivers
 * @return std::vector<specfem::receivers::receiver *> vector of instantiated
 * receiver objects
 */
std::vector<specfem::receivers::receiver *>
read_receivers(const std::string stations_file, const type_real angle);
} // namespace receivers
} // namespace specfem

#endif
