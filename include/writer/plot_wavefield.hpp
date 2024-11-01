#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"
#include "enumerations/wavefield.hpp"
#include "writer.hpp"
#include <boost/filesystem.hpp>

namespace specfem {
namespace writer {
class plot_wavefield : public writer {
public:
  plot_wavefield(const specfem::compute::assembly &assembly,
                 const specfem::display::format &output_format,
                 const specfem::display::wavefield &component,
                 const specfem::wavefield::type &wavefield,
                 const std::string &output_folder)
      : output_format(output_format), component(component),
        output_folder(output_folder), wavefield(wavefield), assembly(assembly) {
  }

  void write() override;

private:
  specfem::display::format output_format;
  specfem::display::wavefield component;
  specfem::wavefield::type wavefield;
  boost::filesystem::path output_folder;
  specfem::compute::assembly assembly;
};
} // namespace writer
} // namespace specfem
