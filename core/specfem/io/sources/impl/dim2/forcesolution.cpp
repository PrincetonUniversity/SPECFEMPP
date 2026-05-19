#include "specfem/io/sources/impl/reader.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim2>>>
specfem::io::sources_impl::read<specfem::element::dimension_tag::dim2,
                                specfem::enums::source_format::FORCESOLUTION>(
    const std::string & /*file_path*/, int /*nsteps*/, type_real /*dt*/,
    specfem::simulation::field_type /*wavefield_type*/) {
  throw std::runtime_error(
      "FORCESOLUTION format is not supported for 2D simulations");
}
