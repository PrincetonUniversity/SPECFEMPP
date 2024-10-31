#pragma once

#include "enumerations/display.hpp"

namespace specfem {
namespace writer {
class plot_wavefield : public writer {
public:
  plot_wavefield(const specfem::compute::assembly &assembly,
                 const specfem::display::format &output_format,
                 const specfem::display::wavefield &wavefield_type,
                 const std::string &output_folder)
      : output_format(output_format), wavefield_type(wavefield_type),
        output_folder(output_folder) {}

private:
  specfem::display::format output_format;
  specfem::display::wavefield wavefield_type;
  std::string output_folder;
};
} // namespace writer
} // namespace specfem
