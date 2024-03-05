#ifndef SPECFEM_WAVEFIELD_WRITER_HPP
#define SPECFEM_WAVEFIELD_WRITER_HPP

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "writer/writer.hpp"

namespace specfem {
namespace writer {

template <typename OutputLibrary> class wavefield : public writer {

public:
  using elastic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::elastic>;
  using acoustic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::acoustic>;

  wavefield(const specfem::compute::assembly &assembly,
            const std::string output_folder);

  void write() override;

private:
  std::string output_folder; ///< Path to output folder
  specfem::compute::impl::field_impl<elastic_type> elastic_field;
  specfem::compute::impl::field_impl<acoustic_type> acoustic_field;
};
} // namespace writer
} // namespace specfem

#endif /* SPECFEM_WAVEFIELD_WRITER_HPP */
